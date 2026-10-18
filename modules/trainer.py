"""Training loop for the AGVS-MNER model."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import optim
from tqdm.auto import tqdm
from transformers.optimization import get_linear_schedule_with_warmup


_AUXILIARY_GROUNDING_PARAMETERS = (
    "grounding_text_proj.",
    "grounding_image_proj.",
    "grounding_none",
    "grounding_fusion_proj.",
)


class NerTrainer:
    """Train, validate, checkpoint, and test a multimodal NER model."""

    def __init__(
        self,
        train_data: Any,
        dev_data: Any,
        test_data: Any,
        model: torch.nn.Module,
        label_map: dict[str, int],
        args: Any,
        logger: logging.Logger,
    ) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.label_map = label_map
        self.index_to_label = {index: label for label, index in label_map.items()}
        self.args = args
        self.logger = logger

        self.train_num_steps = len(train_data) * args.num_epochs
        self.optimizer: optim.Optimizer | None = None
        self.scheduler: Any = None
        self.best_dev_f1 = float("-inf")
        self.best_dev_epoch = 0
        self.patience_counter = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_directory = Path(args.output_dir) / f"{args.dataset.lower()}_{timestamp}"
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self.results_path = self.run_directory / "metrics.csv"
        self.checkpoint_path = self.run_directory / "best_model.pt"
        self._write_config()
        self._initialize_metrics_file()

    def _write_config(self) -> None:
        with (self.run_directory / "config.json").open("w", encoding="utf-8") as config_file:
            json.dump(vars(self.args), config_file, indent=2, ensure_ascii=False)

    def _initialize_metrics_file(self) -> None:
        with self.results_path.open("w", newline="", encoding="utf-8") as metrics_file:
            csv.writer(metrics_file).writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "train_f1",
                    "train_precision",
                    "train_recall",
                    "dev_loss",
                    "dev_accuracy",
                    "dev_f1",
                    "dev_precision",
                    "dev_recall",
                ]
            )

    def _log_epoch_results(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        dev_metrics: dict[str, float],
    ) -> None:
        with self.results_path.open("a", newline="", encoding="utf-8") as metrics_file:
            csv.writer(metrics_file).writerow(
                [
                    epoch,
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['accuracy']:.4f}",
                    f"{train_metrics['f1']:.4f}",
                    f"{train_metrics['precision']:.4f}",
                    f"{train_metrics['recall']:.4f}",
                    f"{dev_metrics['loss']:.6f}",
                    f"{dev_metrics['accuracy']:.4f}",
                    f"{dev_metrics['f1']:.4f}",
                    f"{dev_metrics['precision']:.4f}",
                    f"{dev_metrics['recall']:.4f}",
                ]
            )

    def _move_batch_to_device(self, batch: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            value.to(self.args.device) if isinstance(value, torch.Tensor) else value
            for value in batch
        )

    def _step(
        self, batch: tuple[Any, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], torch.Tensor]:
        input_ids, token_type_ids, attention_mask, image_pixel_values, labels, image_mask = batch
        decoded_label_ids, loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            image_pixel_values=image_pixel_values,
            image_mask=image_mask,
            labels=labels,
        )
        return attention_mask, labels, decoded_label_ids, loss

    def _append_predictions(
        self,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        decoded_label_ids: list[list[int]],
        true_labels: list[list[str]],
        predicted_labels: list[list[str]],
    ) -> None:
        masks = attention_mask.detach().cpu().tolist()
        target_label_ids = labels.detach().cpu().tolist()

        for mask, target_ids, prediction_ids in zip(masks, target_label_ids, decoded_label_ids):
            sequence_length = sum(mask)
            target_ids = target_ids[1:sequence_length]
            prediction_ids = prediction_ids[1:sequence_length]
            filtered_pairs = [
                (target_id, prediction_id)
                for target_id, prediction_id in zip(target_ids, prediction_ids)
                if self.index_to_label[target_id] != "X"
            ]
            sample_target_ids = [target_id for target_id, _ in filtered_pairs]
            outside_label_id = self.label_map["O"]
            subword_label_id = self.label_map.get("X")
            sample_prediction_ids = [
                outside_label_id if prediction_id == subword_label_id else prediction_id
                for _, prediction_id in filtered_pairs
            ]
            true_labels.append([self.index_to_label[label_id] for label_id in sample_target_ids])
            predicted_labels.append(
                [self.index_to_label[label_id] for label_id in sample_prediction_ids]
            )

    def _run_epoch(
        self, data_loader: Any, training: bool, progress_bar: Any = None
    ) -> dict[str, Any]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        loss_component_totals: dict[str, float] = {}
        true_labels: list[list[str]] = []
        predicted_labels: list[list[str]] = []

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in data_loader:
                attention_mask, labels, decoded_label_ids, loss = self._step(
                    self._move_batch_to_device(batch)
                )
                total_loss += loss.detach().item()
                for name, value in getattr(
                    self.model, "last_loss_components", {}
                ).items():
                    scalar = (
                        value.detach().item()
                        if isinstance(value, torch.Tensor)
                        else float(value)
                    )
                    loss_component_totals[name] = (
                        loss_component_totals.get(name, 0.0) + scalar
                    )

                if training:
                    loss.backward()
                    self._clip_gradients()
                    assert self.optimizer is not None
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix(loss=f"{loss.detach().item():.4f}")

                self._append_predictions(
                    attention_mask,
                    labels,
                    decoded_label_ids,
                    true_labels,
                    predicted_labels,
                )

        # Use the same seqeval definitions for logging, early stopping, checkpoint
        # selection, and final reporting. This prevents the displayed micro F1 from
        # disagreeing with the value used to select the best checkpoint.
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        return {
            "loss": total_loss / max(len(data_loader), 1),
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "report": classification_report(
                true_labels,
                predicted_labels,
                digits=4,
                zero_division=0,
            ),
            "loss_components": {
                name: total / max(len(data_loader), 1)
                for name, total in loss_component_totals.items()
            },
        }

    def _clip_gradients(self) -> None:
        """Clip NER and detached teacher-head gradients independently.

        The auxiliary grounding head can have a much larger gradient norm than
        the main task. Including it in one global norm silently scales down the
        otherwise detached BERT/ViT/CRF gradients and breaks the baseline control.
        """
        main_parameters = []
        auxiliary_parameters = []
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            if name.startswith(_AUXILIARY_GROUNDING_PARAMETERS):
                auxiliary_parameters.append(parameter)
            else:
                main_parameters.append(parameter)
        if main_parameters:
            torch.nn.utils.clip_grad_norm_(
                main_parameters, self.args.max_grad_norm
            )
        if auxiliary_parameters:
            torch.nn.utils.clip_grad_norm_(
                auxiliary_parameters, self.args.max_grad_norm
            )

    def _configure_utility_gate_for_epoch(self, epoch: int) -> None:
        if not getattr(self.model, "llm_utility_gate", False):
            return
        warmup_epochs = int(
            getattr(self.args, "llm_utility_gate_warmup_epochs", 2)
        )
        active = epoch > warmup_epochs
        if active and epoch == warmup_epochs + 1:
            parameter = getattr(self.model, "utility_gate_scale", None)
            if parameter is not None:
                with torch.no_grad():
                    parameter.zero_()
        self.model.llm_utility_gate_active = active
        self.logger.info(
            "Utility gate %s at epoch %d (selector warmup: %d epochs)",
            "active" if active else "inactive",
            epoch,
            warmup_epochs,
        )

    def _restore_utility_gate_for_best_checkpoint(self) -> None:
        if not getattr(self.model, "llm_utility_gate", False):
            return
        warmup_epochs = int(
            getattr(self.args, "llm_utility_gate_warmup_epochs", 2)
        )
        self.model.llm_utility_gate_active = self.best_dev_epoch > warmup_epochs

    def train(self) -> None:
        self._configure_optimization()
        self.logger.info("Training outputs will be written to %s", self.run_directory)
        progress_bar = tqdm(total=self.train_num_steps, desc="Training")

        for epoch in range(1, self.args.num_epochs + 1):
            self._configure_utility_gate_for_epoch(epoch)
            self.logger.info("Starting epoch %d/%d", epoch, self.args.num_epochs)
            train_metrics = self._run_epoch(
                self.train_data, training=True, progress_bar=progress_bar
            )
            self.logger.info("Train results:\n%s", train_metrics["report"])
            self.logger.info(
                "Train loss components: %s", train_metrics["loss_components"]
            )

            dev_metrics = self.evaluate(epoch)
            self._log_epoch_results(epoch, train_metrics, dev_metrics)
            learned_controls = {}
            for name in ("grounding_fusion_scale", "utility_gate_scale"):
                parameter = getattr(self.model, name, None)
                if parameter is not None:
                    effective = torch.tanh(parameter.detach())
                    if name == "utility_gate_scale":
                        effective = effective.clamp(min=0.0, max=1.0)
                    learned_controls[name] = {
                        "raw": float(parameter.detach()),
                        "effective": float(effective),
                    }
            if learned_controls:
                self.logger.info("Learned visual controls: %s", learned_controls)
            self.logger.info(
                "Epoch %d: best dev F1 %.4f (epoch %d)",
                epoch,
                self.best_dev_f1,
                self.best_dev_epoch,
            )

            if self.args.patience > 0 and self.patience_counter >= self.args.patience:
                self.logger.info(
                    "Early stopping after %d epochs without improvement.", self.args.patience
                )
                break

        progress_bar.close()
        self.test()

    def evaluate(self, epoch: int) -> dict[str, Any]:
        self.logger.info("Evaluating on the validation set.")
        metrics = self._run_epoch(self.dev_data, training=False)
        self.logger.info("Validation results:\n%s", metrics["report"])
        self.logger.info("Validation loss components: %s", metrics["loss_components"])

        if metrics["f1"] >= self.best_dev_f1:
            self.best_dev_f1 = metrics["f1"]
            self.best_dev_epoch = epoch
            self.patience_counter = 0
            torch.save(self.model.state_dict(), self.checkpoint_path)
            self.logger.info("Saved checkpoint to %s", self.checkpoint_path)
        else:
            self.patience_counter += 1
        return metrics

    def test(self) -> None:
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.args.device))
        self._restore_utility_gate_for_best_checkpoint()
        self.model.to(self.args.device)
        self.logger.info("Evaluating the best checkpoint on the test set.")
        metrics = self._run_epoch(self.test_data, training=False)
        self.logger.info("Test results:\n%s", metrics["report"])
        self.logger.info("Final test F1: %.4f", metrics["f1"])

    def _configure_optimization(self) -> None:
        parameter_groups = {"bert": [], "vit": [], "other": []}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("bert."):
                parameter_groups["bert"].append(parameter)
            elif name.startswith("vit."):
                parameter_groups["vit"].append(parameter)
            else:
                parameter_groups["other"].append(parameter)

        self.optimizer = optim.AdamW(
            [
                {
                    "params": parameter_groups["bert"],
                    "lr": self.args.bert_lr,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": parameter_groups["vit"],
                    "lr": self.args.vit_lr,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": parameter_groups["other"],
                    "lr": self.args.lr,
                    "weight_decay": self.args.weight_decay,
                },
            ]
        )
        self.model.to(self.args.device)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * self.train_num_steps),
            num_training_steps=self.train_num_steps,
        )
