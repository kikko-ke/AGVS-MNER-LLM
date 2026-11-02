#!/usr/bin/env bash
# Never install VLM dependencies into the working BERT/ViT base environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /root/autodl-tmp/.vlm_install_tmp
export TMPDIR=/root/autodl-tmp/.vlm_install_tmp
envdir=/root/autodl-tmp/envs/mner-vlm
if [[ ! -x "$envdir/bin/python" ]]; then
  python - <<'PY'
import shutil
free = shutil.disk_usage('/root/autodl-tmp').free / 1024**3
print(f'Data-disk free space: {free:.1f} GiB')
if free < 20:
    raise SystemExit('Allow about 20 GiB for the new environment, model and install temporaries. Preserve old results and check storage first.')
PY
  conda create --prefix "$envdir" python=3.11 -y
fi
"$envdir/bin/python" -m pip install --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$envdir/bin/python" -m pip install --no-cache-dir -r requirements-vlm.txt
"$envdir/bin/python" -m pip check
"$envdir/bin/python" - <<'PY'
from modules.vlm_training import check_runtime
print('Pinned VLM environment ready:', check_runtime())
PY
echo "Activate with: conda activate $envdir"
