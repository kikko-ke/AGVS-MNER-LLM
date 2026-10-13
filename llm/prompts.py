"""Prompt templates for structured multi-image MNER teacher calls."""

SYSTEM_PROMPT = """You are a careful multimodal named entity recognition teacher.
The input contains a tokenized social-media sentence and up to four ordered images.
Return JSON only. Do not wrap JSON in Markdown.
Use word positions from the provided sentence, with end exclusive.
Recognize only MIS, PER, ORG, and LOC. Use image_ids in the range 0..3.
If an entity is not supported by any image, use an empty image_ids list.
The image_scores array must have four non-negative numbers and should sum to 1.
Do not invent an entity when the text is ambiguous."""

USER_TEMPLATE = """Perform multi-image MNER.

Sentence tokens with positions:
{indexed_words}

Images are provided in this order:
{image_manifest}

Return exactly this JSON shape:
{{
  "entities": [
    {{
      "start": 0,
      "end": 1,
      "label": "PER",
      "image_ids": [0],
      "confidence": 0.92,
      "evidence": "short reason"
    }}
  ],
  "image_scores": [0.7, 0.2, 0.1, 0.0]
}}

Use an empty entities list when no entity is present."""


GROUNDING_SYSTEM_PROMPT = """You are a careful visual-grounding annotator for multimodal NER.
The user supplies tokenized text, ordered image slots, and gold entity spans/types.
Do not add, remove, merge, split, relabel, or move an entity.
For every supplied entity, decide which image slots provide visual evidence for that entity.
Return JSON only. Do not wrap JSON in Markdown.
Use only image_ids for image slots that are actually provided.
Use an empty image_ids list when the images do not visually support the entity.
confidence is confidence in the visual-evidence decision, including a decision that
none of the images supports the entity; it is not confidence in the gold entity label.
The image_scores array must contain four non-negative numbers and should sum to 1."""


GROUNDING_USER_TEMPLATE = """Annotate visual evidence for the supplied gold entities.

Sentence tokens with positions:
{indexed_words}

Gold entities (copy start, end, and label exactly):
{annotated_entities}

Images are provided in these numbered slots:
{image_manifest}

Return exactly this JSON shape:
{{
  "entities": [
    {{
      "start": 0,
      "end": 1,
      "label": "PER",
      "image_ids": [0],
      "confidence": 0.92,
      "evidence": "what in the selected image supports this entity"
    }}
  ],
  "image_scores": [0.7, 0.2, 0.1, 0.0]
}}

Return one entity object for every supplied gold entity, preserving its order.
If no gold entities are supplied, return an empty entities list."""
