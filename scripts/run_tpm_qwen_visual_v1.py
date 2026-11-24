"""Compare existing text-only Qwen features with new image-conditioned features."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.tpm_qwen_visual_v1.pipeline import main

if __name__ == "__main__":
    main()
