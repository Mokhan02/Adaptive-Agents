#!/usr/bin/env bash
# Set up a Lambda Cloud instance for the in-context sweep.
#
#   curl -fsSL https://raw.githubusercontent.com/Mokhan02/Adaptive-Agents/main/scripts/lambda_setup.sh | bash
#   # or, after cloning: bash scripts/lambda_setup.sh
#
# Uses the instance's preinstalled PyTorch (Lambda Stack); installs only the
# light dependencies. Then start the sweep inside tmux so it survives a
# dropped SSH connection:
#
#   cd ~/Adaptive-Agents && tmux new -s sweep
#   PYTHONPATH=src python3 scripts/tune_icl.py --parallel 8 2>&1 | tee sweep.log
#   # detach: Ctrl-b d    reattach: tmux attach -t sweep
set -euo pipefail

cd ~
[ -d Adaptive-Agents ] || git clone https://github.com/Mokhan02/Adaptive-Agents.git
cd Adaptive-Agents
git pull --ff-only

python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
python3 -m pip install --user -q "numpy>=1.25,<2" "matplotlib>=3.7" pytest  # spawn needs 1.25; prebuilt torch needs <2
python3 -m pytest -q tests/test_in_context.py tests/test_pretrain.py
echo "vCPUs: $(nproc)"
echo "Ready. Start the sweep in tmux (see the top of this script)."
