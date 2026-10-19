from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from scripts.evaluate_visual_ablation import (
    ImageMaskOverrideDataset,
    ShuffledImageDataset,
)


class _SingleSampleDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor([101, 102]),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            torch.ones((4, 3, 2, 2)),
            torch.zeros(2, dtype=torch.long),
            torch.tensor([1, 1, 0, 0]),
        )


def test_text_only_ablation_changes_only_the_image_mask() -> None:
    base = _SingleSampleDataset()
    original = base[0]
    ablated = ImageMaskOverrideDataset(base)[0]

    assert len(ablated) == len(original)
    for original_value, ablated_value in zip(original[:-1], ablated[:-1]):
        assert torch.equal(original_value, ablated_value)
    assert ablated[-1].tolist() == [0, 0, 0, 0]


class _ShuffleDataset:
    def __init__(self) -> None:
        self.masks = (
            torch.tensor([1, 0, 0, 0]),
            torch.tensor([1, 0, 0, 0]),
            torch.tensor([1, 1, 0, 0]),
            torch.tensor([1, 1, 0, 0]),
        )

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor([101, 102]),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            torch.full((4, 3, 2, 2), float(index)),
            torch.zeros(2, dtype=torch.long),
            self.masks[index],
        )


def test_shuffled_images_preserve_masks_and_replace_content() -> None:
    base = _ShuffleDataset()
    shuffled = ShuffledImageDataset(base, seed=1234)

    assert shuffled.changed_samples == len(base)
    for index in range(len(base)):
        sample = shuffled[index]
        donor_index = shuffled.donor_indices[index]
        assert donor_index != index
        assert torch.equal(sample[-1], base[index][-1])
        assert torch.equal(sample[3], base[donor_index][3])
        assert torch.equal(base[index][-1], base[donor_index][-1])


def test_visual_ablation_script_runs_directly() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(project_root / "scripts" / "evaluate_visual_ablation.py"),
            "--help",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
