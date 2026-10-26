"""Training adapter that feeds cached teacher signals into the AGVS model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from modules.trainer import NerTrainer


def build_evidence_route_mask(
    image_mask: torch.Tensor,
    grounding_targets: torch.Tensor,
    grounding_weights: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    outside_label_id: int,
    subword_label_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the union of teacher evidence images for supervised entities.

    Samples without a teacher-supervised entity retain their full image mask.
    A supervised NONE-only sample receives a text-only routed view.  The final
    tensor contains confidence/class weights only at entity positions whose
    routed image mask actually differs from the full view.
    """
    if grounding_targets.ndim != 3 or grounding_weights.ndim != 2:
        raise ValueError("Grounding targets/weights must be batched token tensors")
    if grounding_targets.shape[:2] != grounding_weights.shape:
        raise ValueError("Grounding target and weight token dimensions must match")
    if labels.shape != grounding_weights.shape or attention_mask.shape != labels.shape:
        raise ValueError("Labels, attention mask, and grounding weights must match")

    image_mask_bool = image_mask.bool()
    num_images = image_mask_bool.shape[-1]
    entity_positions = (grounding_weights > 0) & attention_mask.bool()
    entity_positions &= labels != outside_label_id
    if subword_label_id is not None:
        entity_positions &= labels != subword_label_id

    selected_slots = grounding_targets[..., :num_images] > 0
    evidence_union = (selected_slots & entity_positions.unsqueeze(-1)).any(dim=1)
    evidence_union &= image_mask_bool
    has_supervised_entity = entity_positions.any(dim=1)

    routed_mask = image_mask_bool.clone()
    routed_mask[has_supervised_entity] = evidence_union[has_supervised_entity]
    route_changed = has_supervised_entity & (routed_mask != image_mask_bool).any(dim=1)
    routed_token_weights = (
        grounding_weights
        * entity_positions.to(grounding_weights.dtype)
        * route_changed.unsqueeze(1).to(grounding_weights.dtype)
    )
    return routed_mask.to(image_mask.dtype), route_changed, routed_token_weights


def routed_emission_kl(
    full_emissions: torch.Tensor,
    routed_emissions: torch.Tensor,
    token_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Distill detached routed emissions into full-view token emissions."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if full_emissions.shape != routed_emissions.shape:
        raise ValueError("Full and routed emissions must have the same shape")
    if token_weights.shape != full_emissions.shape[:2]:
        raise ValueError("Token weights must match emission batch/token dimensions")

    teacher_probabilities = torch.softmax(
        routed_emissions.detach() / temperature,
        dim=-1,
    )
    token_kl = F.kl_div(
        F.log_softmax(full_emissions / temperature, dim=-1),
        teacher_probabilities,
        reduction="none",
    ).sum(dim=-1)
    weights = token_weights.to(token_kl.dtype)
    return (
        (token_kl * weights).sum()
        / weights.sum().clamp_min(1.0)
        * (temperature ** 2)
    )


def routed_visual_strength(
    full_image_mask: torch.Tensor,
    routed_image_mask: torch.Tensor,
) -> torch.Tensor:
    """Preserve full-view visual scale when the routed view keeps evidence.

    AGVS normally scales visual fusion by the number of valid images.  A route
    that removes distractors should change image content without also shrinking
    the useful signal merely because fewer slots remain.  Explicit NONE routes
    still receive zero visual strength.
    """
    if full_image_mask.shape != routed_image_mask.shape:
        raise ValueError("Full and routed image masks must have the same shape")
    if full_image_mask.ndim != 2:
        raise ValueError("Image masks must have shape [batch, image_count]")
    full_fraction = full_image_mask.bool().float().mean(dim=1)
    has_routed_evidence = routed_image_mask.bool().any(dim=1).float()
    return full_fraction * has_routed_evidence


class LlmNerTrainer(NerTrainer):
    def _step(
        self, batch: tuple[Any, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], torch.Tensor]:
        if len(batch) not in {11, 15}:
            raise ValueError(f"Unexpected LLM batch width: {len(batch)}")
        base_batch = batch[:11]
        (
            input_ids,
            token_type_ids,
            attention_mask,
            image_pixel_values,
            labels,
            image_mask,
            llm_token_probs,
            llm_image_scores,
            llm_available,
            llm_grounding_targets,
            llm_grounding_weights,
        ) = base_batch
        evidence_batch = batch[11:] if len(batch) == 15 else (None,) * 4
        (
            llm_evidence_token_masks,
            llm_evidence_image_targets,
            llm_evidence_embeddings,
            llm_evidence_weights,
        ) = evidence_batch
        routed_crf_weight = float(
            getattr(self.args, "llm_routed_crf_loss_weight", 0.0)
        )
        consistency_weight = float(
            getattr(self.args, "llm_consistency_kl_weight", 0.0)
        )
        routed_losses_enabled = self.model.training and (
            routed_crf_weight > 0 or consistency_weight > 0
        )
        routed_image_mask = image_mask
        changed_samples = torch.zeros(
            image_mask.shape[0], dtype=torch.bool, device=image_mask.device
        )
        routed_token_weights = torch.zeros_like(llm_grounding_weights)
        if routed_losses_enabled:
            routed_image_mask, changed_samples, routed_token_weights = (
                build_evidence_route_mask(
                    image_mask,
                    llm_grounding_targets,
                    llm_grounding_weights,
                    labels,
                    attention_mask,
                    outside_label_id=self.label_map["O"],
                    subword_label_id=self.label_map.get("X"),
                )
            )
        use_routed_view = routed_losses_enabled and bool(changed_samples.any())

        # Replay identical dropout/image-dropout randomness for the two views.
        # Their only intended difference is the evidence-preserving image mask.
        cpu_rng_before = torch.get_rng_state() if use_routed_view else None
        cuda_rng_before = (
            torch.cuda.get_rng_state_all()
            if use_routed_view and image_pixel_values.is_cuda
            else None
        )
        full_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            image_pixel_values=image_pixel_values,
            image_mask=image_mask,
            labels=labels,
            llm_token_probs=llm_token_probs,
            llm_image_scores=llm_image_scores,
            llm_available=llm_available,
            llm_grounding_targets=llm_grounding_targets,
            llm_grounding_weights=llm_grounding_weights,
            llm_evidence_token_masks=llm_evidence_token_masks,
            llm_evidence_image_targets=llm_evidence_image_targets,
            llm_evidence_embeddings=llm_evidence_embeddings,
            llm_evidence_weights=llm_evidence_weights,
            return_features=use_routed_view,
        )
        if use_routed_view:
            decoded_label_ids, loss, full_features = full_outputs
        else:
            decoded_label_ids, loss = full_outputs
            if routed_losses_enabled:
                components = dict(
                    getattr(self.model, "last_loss_components", {})
                )
                components["evidence_route_fraction"] = torch.zeros(
                    (), device=image_mask.device
                )
                self.model.last_loss_components = components
            return attention_mask, labels, decoded_label_ids, loss

        components = dict(getattr(self.model, "last_loss_components", {}))
        components["evidence_route_fraction"] = changed_samples.float().mean().detach()
        cpu_rng_after = torch.get_rng_state()
        cuda_rng_after = (
            torch.cuda.get_rng_state_all() if image_pixel_values.is_cuda else None
        )
        active = changed_samples.nonzero(as_tuple=False).squeeze(-1)
        routed_arguments = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "image_pixel_values": image_pixel_values,
            "image_mask": routed_image_mask,
            "visual_strength_override": routed_visual_strength(
                image_mask,
                routed_image_mask,
            ),
            "labels": None,
            "return_features": True,
        }
        assert cpu_rng_before is not None
        torch.set_rng_state(cpu_rng_before)
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state_all(cuda_rng_before)
        try:
            if routed_crf_weight > 0:
                _, _, routed_features = self.model(**routed_arguments)
            else:
                with torch.no_grad():
                    _, _, routed_features = self.model(**routed_arguments)
        finally:
            torch.set_rng_state(cpu_rng_after)
            if cuda_rng_after is not None:
                torch.cuda.set_rng_state_all(cuda_rng_after)

        routed_emissions_all = routed_features["token_emissions"]
        routed_emissions = routed_emissions_all.index_select(0, active)
        full_emissions = full_features["token_emissions"].index_select(0, active)

        if routed_crf_weight > 0:
            routed_sequence_nll = -self.model.crf(
                routed_emissions_all,
                labels,
                mask=attention_mask.bool(),
                reduction="none",
            )
            routed_crf_loss = routed_sequence_nll[changed_samples].mean()
            loss = loss + routed_crf_weight * routed_crf_loss
            components["routed_crf"] = routed_crf_loss.detach()

        if consistency_weight > 0:
            consistency_loss = routed_emission_kl(
                full_emissions,
                routed_emissions,
                routed_token_weights.index_select(0, active),
                float(getattr(self.args, "llm_consistency_temperature", 1.0)),
            )
            loss = loss + consistency_weight * consistency_loss
            components["routed_emission_kl"] = consistency_loss.detach()

        self.model.last_loss_components = components
        return attention_mask, labels, decoded_label_ids, loss
