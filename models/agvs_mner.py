import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel
from torchcrf import CRF


class RelativePositionMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        max_relative_position=3,
        dropout=0.3,
        batch_first=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.max_relative_position = max_relative_position

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        num_relative_positions = 2 * max_relative_position + 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )
        nn.init.normal_(self.relative_position_bias_table, std=0.02)

    def _generate_relative_position_index(self, target_length, device):
        range_vec = torch.arange(target_length, device=device)
        distance_matrix = range_vec[None, :] - range_vec[:, None]
        distance_matrix_clipped = torch.clamp(
            distance_matrix,
            -self.max_relative_position,
            self.max_relative_position
        )
        return distance_matrix_clipped + self.max_relative_position

    def forward(self, query, key, value, key_padding_mask=None):
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        target_length, batch_size, embed_dim = query.shape
        source_length = key.shape[0]
        device = query.device

        relative_position_index = self._generate_relative_position_index(target_length, device)

        q = self.q_proj(query).view(
            target_length, batch_size * self.num_heads, self.head_dim
        ).transpose(0, 1)
        k = self.k_proj(key).view(
            source_length, batch_size * self.num_heads, self.head_dim
        ).transpose(0, 1)
        v = self.v_proj(value).view(
            source_length, batch_size * self.num_heads, self.head_dim
        ).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(-2, -1))
        attn_output_weights = attn_output_weights / (self.head_dim ** 0.5)

        if target_length == source_length:
            relative_position_bias = self.relative_position_bias_table[
                relative_position_index.view(-1)
            ].view(target_length, target_length, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
            relative_position_bias = relative_position_bias.expand(
                batch_size, -1, -1, -1
            ).reshape(batch_size * self.num_heads, target_length, target_length)

            attn_output_weights = attn_output_weights + relative_position_bias

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.bool()
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            key_padding_mask = key_padding_mask.expand(
                -1, self.num_heads, target_length, source_length
            )
            key_padding_mask = key_padding_mask.reshape(
                batch_size * self.num_heads, target_length, source_length
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask,
                torch.finfo(attn_output_weights.dtype).min
            )

        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            target_length, batch_size, embed_dim
        )
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights


class RelativePositionTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        max_relative_position=3
    ):
        super().__init__()
        self.self_attn = RelativePositionMultiheadAttention(
            d_model,
            nhead,
            max_relative_position=max_relative_position,
            dropout=dropout,
            batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        del src_mask
        attention_output = self.self_attn(
            src,
            src,
            src,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(attention_output)
        src = self.norm1(src)

        feed_forward_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(feed_forward_output)
        src = self.norm2(src)

        return src


class AgvsMnerModel(nn.Module):
    """AGVS-MNER with the original numerical architecture and parameter order."""
    def __init__(self, label_list, args):
        super().__init__()
        self.args = args
        self.num_labels = len(label_list)
        self.hidden_size = 768
        self.num_images = 4
        self.num_patches_per_image = 196
        self.topk_img_patches = getattr(args, "topk_img_patches", 12)
        self.image_dropout_prob = getattr(args, "image_dropout_prob", 0.2)
        self.llm_token_loss_weight = getattr(args, "llm_token_loss_weight", 0.0)
        self.llm_image_loss_weight = getattr(args, "llm_image_loss_weight", 0.0)

        # Text encoder
        self.bert = BertModel.from_pretrained(args.bert_model)

        # Image encoder
        self.vit = ViTModel.from_pretrained(args.vit_model)

        # Freeze ViT: train encoder layers 6-11 only
        for name, param in self.vit.named_parameters():
            if "encoder.layer" in name:
                layer_id = int(name.split("encoder.layer.")[1].split(".")[0])
                param.requires_grad = layer_id >= 6
            else:
                param.requires_grad = False

        # Kept under its original state-dict key for checkpoint compatibility.
        self.temporalEmbedding = nn.Embedding(self.num_images, self.hidden_size)

        # Image sequence encoder
        self.img_seq_encoder = nn.ModuleList([
            RelativePositionTransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                max_relative_position=3
            )
            for _ in range(2)
        ])

        # Query-based global image aggregation
        self.img_global_query = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        nn.init.normal_(self.img_global_query, std=0.02)

        self.text_to_global_query = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )

        self.img_global_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        self.img_global_norm = nn.LayerNorm(self.hidden_size)
        self.img_global_dropout = nn.Dropout(0.1)

        # These frozen parameter blocks are intentionally retained. They preserve the
        # original initialization order and checkpoint layout, so removing them would
        # change seeded training runs despite their not being used in ``forward``.
        self.text_proj = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(10.0)))

        for parameter in self.text_proj.parameters():
            parameter.requires_grad = False
        for parameter in self.image_proj.parameters():
            parameter.requires_grad = False
        self.logit_scale.requires_grad = False

        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.output_dropout = nn.Dropout(getattr(args, "output_dropout", 0.2))

        # Channel-wise patch gating
        self.channel_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )

        # Vector gate
        self.img_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # NER head
        self.fc = nn.Linear(self.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    @property
    def slot_embedding(self):
        """The paper's learnable image-slot embedding table."""
        return self.temporalEmbedding

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        image_pixel_values=None,
        image_mask=None,
        labels=None,
        llm_token_probs=None,
        llm_image_scores=None,
        llm_available=None,
        return_features=False
    ):
        if image_pixel_values is None:
            raise ValueError("image_pixel_values must not be None")

        batch_size, image_count, channels, height, width = image_pixel_values.shape
        if image_count != self.num_images:
            raise ValueError(f"Expected {self.num_images} images, got {image_count}")

        device = image_pixel_values.device

        # -----------------------------
        # Safe masks
        # -----------------------------
        if attention_mask is None:
            attention_mask = torch.ones_like(
                input_ids,
                dtype=torch.long,
                device=input_ids.device
            )
        attention_mask_bool = attention_mask.bool()

        if image_mask is None:
            image_mask = torch.ones(
                batch_size,
                image_count,
                dtype=torch.long,
                device=device
            )
        image_mask_bool = image_mask.bool()
        real_image_count = image_mask_bool.sum(dim=1, keepdim=True).float()

        safe_image_mask = image_mask_bool.clone()
        all_invalid = safe_image_mask.sum(dim=1) == 0
        if all_invalid.any():
            safe_image_mask[all_invalid, 0] = True

        # -----------------------------
        # 1. Text Encoding
        # -----------------------------
        text_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        text_seq = text_output.last_hidden_state
        text_cls = text_output.pooler_output

        # -----------------------------
        # 2. Image Encoding
        # -----------------------------
        pixel_values = image_pixel_values.view(batch_size * image_count, channels, height, width)
        vit_output = self.vit(pixel_values=pixel_values)
        vit_seq = vit_output.last_hidden_state

        cls_tokens = vit_seq[:, 0, :].reshape(
            batch_size,
            image_count,
            self.hidden_size
        )

        patch_tokens = vit_seq[:, 1:, :].reshape(
            batch_size,
            image_count * self.num_patches_per_image,
            self.hidden_size
        )

        # Add temporal embedding, then remove invalid image tokens.
        temporal_position_ids = torch.arange(image_count, device=device).unsqueeze(0).expand(
            batch_size, -1
        )
        temporal_embeddings = self.slot_embedding(temporal_position_ids)

        cls_tokens = cls_tokens + temporal_embeddings
        cls_tokens = cls_tokens * safe_image_mask.unsqueeze(-1).float()

        # Sequence modeling over image CLS tokens.
        # Important for single-image samples: padding image positions must not attend.
        image_key_padding_mask = ~safe_image_mask
        for layer in self.img_seq_encoder:
            cls_tokens = layer(
                cls_tokens,
                src_key_padding_mask=image_key_padding_mask
            )
            cls_tokens = cls_tokens * safe_image_mask.unsqueeze(-1).float()

        # -----------------------------
        # 3. Text-conditioned global query
        # -----------------------------
        base_query = self.img_global_query.expand(batch_size, -1, -1)
        text_query = self.text_to_global_query(text_cls).unsqueeze(1)
        global_query = base_query + text_query

        V_global_token, global_attn_weights = self.img_global_attn(
            query=global_query,
            key=cls_tokens,
            value=cls_tokens,
            key_padding_mask=image_key_padding_mask
        )

        V_global = global_query + self.img_global_dropout(V_global_token)
        V_global = self.img_global_norm(V_global)
        V_global = V_global.squeeze(1)

        # -----------------------------
        # 4. Temporal embedding for patch tokens
        # -----------------------------
        temporal_ids_for_patches = torch.arange(
            image_count,
            device=device
        ).repeat_interleave(self.num_patches_per_image)

        temporal_embeddings_for_patches = self.slot_embedding(
            temporal_ids_for_patches
        ).unsqueeze(0).expand(batch_size, -1, -1)

        patch_valid_mask = safe_image_mask.unsqueeze(-1).expand(
            -1,
            -1,
            self.num_patches_per_image
        ).reshape(batch_size, image_count * self.num_patches_per_image)

        patch_tokens = patch_tokens + temporal_embeddings_for_patches
        patch_tokens = patch_tokens * patch_valid_mask.unsqueeze(-1).float()

        # -----------------------------
        # 5. Patch selection
        # -----------------------------
        gated_patches = patch_tokens * self.channel_proj(patch_tokens)
        gated_patches = gated_patches * patch_valid_mask.unsqueeze(-1).float()

        global_scores = torch.bmm(
            V_global.unsqueeze(1),
            gated_patches.transpose(-2, -1)
        ).squeeze(1) / (self.hidden_size ** 0.5)

        global_scores = global_scores.masked_fill(
            ~patch_valid_mask,
            torch.finfo(global_scores.dtype).min
        )

        topk = min(self.topk_img_patches, global_scores.size(-1))
        _, topk_indices = torch.topk(global_scores, k=topk, dim=-1)

        batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1)
        selected_patches = gated_patches[batch_indices, topk_indices]

        image_tokens = torch.cat(
            [V_global.unsqueeze(1), selected_patches],
            dim=1
        )

        patch_img_ids = topk_indices // self.num_patches_per_image
        selected_patch_mask = safe_image_mask[batch_indices, patch_img_ids]

        image_tokens_mask = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            selected_patch_mask
        ], dim=1)

        selected_image_key_padding_mask = ~image_tokens_mask

        # -----------------------------
        # 6. Cross-modal attention
        # -----------------------------
        cv, _ = self.cross_modal_attention(
            query=text_seq,
            key=image_tokens,
            value=image_tokens,
            key_padding_mask=selected_image_key_padding_mask
        )

        # -----------------------------
        # 7. Vector gating fusion
        # -----------------------------
        V_global_expanded = V_global.unsqueeze(1).expand(
            -1,
            text_seq.size(1),
            -1
        )

        gate_input = torch.cat([text_seq, V_global_expanded], dim=-1)
        gate = torch.sigmoid(self.img_gate(gate_input))

        visual_strength = (real_image_count / float(self.num_images)).unsqueeze(-1)
        if self.training and self.image_dropout_prob > 0:
            keep_visual = (
                torch.rand(batch_size, 1, 1, device=device) >= self.image_dropout_prob
            ).float()
            visual_strength = visual_strength * keep_visual

        fused_text = text_seq + visual_strength * gate * cv
        fused_text = self.layer_norm(fused_text)
        fused_text = self.output_dropout(fused_text)

        # -----------------------------
        # 8. NER head
        # -----------------------------
        fused_text = fused_text * attention_mask.unsqueeze(-1).float()
        emissions = self.fc(fused_text)

        if labels is not None:
            ner_loss = -self.crf(
                emissions,
                labels,
                mask=attention_mask_bool,
                reduction="mean"
            )
            total_loss = ner_loss

            if llm_token_probs is not None and self.llm_token_loss_weight > 0:
                teacher_probs = llm_token_probs.to(emissions.device).clamp_min(1e-6)
                teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)
                token_kl = F.kl_div(
                    F.log_softmax(emissions, dim=-1),
                    teacher_probs,
                    reduction="none",
                ).sum(dim=-1)
                valid_tokens = attention_mask_bool.float()
                if llm_available is not None:
                    valid_tokens = valid_tokens * llm_available.float().view(-1, 1)
                total_loss = total_loss + self.llm_token_loss_weight * (
                    token_kl * valid_tokens
                ).sum() / valid_tokens.sum().clamp_min(1.0)

            if llm_image_scores is not None and self.llm_image_loss_weight > 0:
                student_images = global_attn_weights.squeeze(1).clamp_min(1e-6)
                student_images = student_images / student_images.sum(dim=-1, keepdim=True)
                teacher_images = llm_image_scores.to(emissions.device).clamp_min(1e-6)
                teacher_images = teacher_images / teacher_images.sum(dim=-1, keepdim=True)
                image_kl = F.kl_div(
                    student_images.log(),
                    teacher_images,
                    reduction="none",
                ).sum(dim=-1)
                valid_samples = (
                    llm_available.float()
                    if llm_available is not None
                    else torch.ones_like(image_kl)
                )
                total_loss = total_loss + self.llm_image_loss_weight * (
                    image_kl * valid_samples
                ).sum() / valid_samples.sum().clamp_min(1.0)

            logits = self.crf.decode(
                emissions,
                mask=attention_mask_bool
            )
            if return_features:
                return logits, total_loss, {
                    "token_emissions": emissions,
                    "image_attention": global_attn_weights,
                }
            return logits, total_loss

        logits = self.crf.decode(
            emissions,
            mask=attention_mask_bool
        )
        if return_features:
            return logits, None, {
                "token_emissions": emissions,
                "image_attention": global_attn_weights,
            }
        return (logits,)



