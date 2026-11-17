"""Independent TPM-MI + frozen BERT/Qwen experiments in the existing repository."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.tpm_llm_v1.pipeline import main

if __name__ == "__main__":
    main()
