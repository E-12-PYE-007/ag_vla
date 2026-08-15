#!/bin/bash

# Set error handling for bash
set -euo pipefail
# e: exit immediately if any command returns a non-zero exit code, instead of silently failing
# u: treat unset variables as errors, instead of silently expanding to empty strings
# o pipefail: if any command in a pipeline (passing the output of one as input to the next by |) fails, the whole pipeline fails

# Set environment variables
PROJECT_ID=capstone
PROJECT_DIR="/data/gpfs/projects/capstone/E12PYE007"
VENV_DIR="${PROJECT_DIR}/venvs/training/finetune"

PYTHON_MODULE="GCCcore/11.3.0 Python/3.10.4"

CUDA_MODULE="CUDA/12.1.1"

OUT_DIR="${PROJECT_DIR}/out"
HF_HOME="${PROJECT_DIR}/.cache/huggingface"
ASYNCVLA_DIR="${PROJECT_DIR}/ag_vla/AsyncVLA"
VISUALNAV_DIR="${ASYNCVLA_DIR}/visualnav-transformer/train"

# Helper functions for terminal output
PASS=0
FAIL=0
ok()   { echo "[OK]   $1"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }
info() { echo "[INFO] $1"; }

# Environment setup
info "Loading modules..."
module purge
module load ${PYTHON_MODULE}

info "Checking virtualenv..."
if [ -f "${VENV_DIR}/bin/activate" ]; then
    ok "Virtualenv exists: ${VENV_DIR}"
else
    info "Virtualenv not found — creating it now..."
    python -m venv "${VENV_DIR}"
    ok "Created virtualenv: ${VENV_DIR}"
fi

info "Activating virtualenv..."
source "${VENV_DIR}/bin/activate"

# Install core dependencies
info "Installing pinned PyTorch..."
pip install \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet
ok "PyTorch 2.2.0 (cu121) installed"

info "Installing AsyncVLA base requirements..."
grep -v "OmniVLA\|vint_train" "${ASYNCVLA_DIR}/requirements.txt" | pip install -r /dev/stdin --quiet
ok "AsyncVLA requirements installed"

# Install editable packages
info "Installing AsyncVLA as editable package..."
pip install -e "${ASYNCVLA_DIR}" --quiet
ok "AsyncVLA installed as editable: ${ASYNCVLA_DIR}"

info "Installing visualnav-transformer as editable package..."
pip install -e "${VISUALNAV_DIR}" --quiet
ok "visualnav-transformer installed as editable: ${VISUALNAV_DIR}"

# Install Flash Attention
info "Loading CUDA toolkit for FlashAttention compilation..."
module load ${CUDA_MODULE}

CUDA_HOME_CANDIDATE=$(dirname "$(dirname "$(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)")")
if [ -d "${CUDA_HOME_CANDIDATE}" ]; then
    export CUDA_HOME="${CUDA_HOME_CANDIDATE}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    info "CUDA_HOME set to: ${CUDA_HOME}"
fi

info "Building FlashAttention 2.5.5..."
pip install flash-attn==2.5.5 --no-build-isolation --quiet \
    && ok "FlashAttention 2.5.5 installed" \
    || { fail "FlashAttention build failed"; }

info "Downloading AsyncVLA_release snapshot from HuggingFace (~16 GB)..."
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
python -c "from huggingface_hub import snapshot_download; snapshot_download('NHirose/AsyncVLA_release')" \
    && ok "AsyncVLA_release snapshot downloaded to ${HF_HOME}" \
    || { fail "AsyncVLA_release snapshot download failed"; }


# Check all required paths exist
info "Checking project storage..."
if [ -d "${PROJECT_DIR}" ]; then
    ok "Project directory exists: ${PROJECT_DIR}"
else
    fail "Project directory not found: ${PROJECT_DIR}"
fi

info "Checking storage quota..."
lfs quota -h -g "${PROJECT_ID}" /data/gpfs/projects/ 2>/dev/null || info "Could not retrieve quota (non-fatal)"

info "Checking Mediaflux setup..."
if [ -f "${HOME}/.Arcitecta/mflux.cfg" ]; then
    ok "Mediaflux config found: ~/.Arcitecta/mflux.cfg"
else
    fail "Mediaflux config not found — run unimelb-mf-login first"
fi

info "Checking output directory..."
if [ -d "${OUT_DIR}" ]; then
    ok "Output directory exists: ${OUT_DIR}"
else
    info "Output directory does not exist — will be created at training time: ${OUT_DIR}"
fi

info "Checking HF cache directory..."
if [ -d "${HF_HOME}" ]; then
    ok "HF cache directory exists: ${HF_HOME}"
else
    info "HF cache directory does not exist — creating now..."
    mkdir -p "${HF_HOME}"
    ok "Created: ${HF_HOME}"
fi

info "Checking AsyncVLA submodule..."
if [ -d "${ASYNCVLA_DIR}/prismatic" ]; then
    ok "AsyncVLA submodule present: ${ASYNCVLA_DIR}"
    SUBMODULE_URL=$(git -C "${ASYNCVLA_DIR}" remote get-url origin 2>/dev/null || echo "unknown")
    if echo "${SUBMODULE_URL}" | grep -q "lyamatomato/AsyncVLA"; then
        ok "AsyncVLA submodule points to correct fork: ${SUBMODULE_URL}"
    else
        fail "AsyncVLA submodule URL looks wrong: ${SUBMODULE_URL}"
        info "Expected: https://github.com/lyamatomato/AsyncVLA.git"
        info "Check .gitmodules and re-run: git submodule sync && git submodule update --init"
    fi
else
    fail "AsyncVLA submodule missing or not initialised: ${ASYNCVLA_DIR}"
    info "Run: git submodule update --init"
fi

# Check all Python imports exist and versions are correct
info "Checking key Python imports..."
python -c "import torch; print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" \
    && ok "torch importable" \
    || fail "torch not importable"

python -c "import draccus, transformers, peft, wandb" \
    && ok "draccus, transformers, peft, wandb importable" \
    || fail "one or more of draccus/transformers/peft/wandb not importable"

python -c "import prismatic" \
    && ok "prismatic importable (AsyncVLA editable install working)" \
    || fail "prismatic not importable — editable install of AsyncVLA may have failed"

python -c "import vint_train" \
    && ok "vint_train importable (visualnav-transformer editable install working)" \
    || fail "vint_train not importable — editable install of visualnav-transformer may have failed"

python -c "import flash_attn; print(f'  flash_attn {flash_attn.__version__}')" \
    && ok "flash_attn importable" \
    || fail "flash_attn not importable — FlashAttention build may have failed"


# Summary
echo ""
echo "────────────────────────────────────────"
echo "  ${PASS} passed   ${FAIL} failed"
echo "────────────────────────────────────────"

if [ "${FAIL}" -gt 0 ]; then
    echo "Fix the above failures before submitting the job."
    exit 1
else
    echo "All checks passed — ready to submit."
    echo "Run: sbatch one of training/finetune/ag-vla-finetune-<type>.slurm"
    exit 0
fi