"""Download open model weights once; this does not query a teacher API."""
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/root/autodl-tmp/pretrained_models/Qwen2.5-VL-3B-Instruct')
    args = parser.parse_args()
    print(snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', local_dir=args.output, max_workers=4,
          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']))

if __name__ == '__main__':
    main()
