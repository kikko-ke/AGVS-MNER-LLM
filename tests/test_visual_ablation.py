from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from scripts.evaluate_visual_ablation import ImageMaskOverrideDataset


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
