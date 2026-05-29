# Sourced by every sbatch script. Keeps env, paths, and uv setup in one place.
# Do not run directly. Do not `set -e` here — callers may want to handle failures themselves.

# --- Paths ---
export RELP_REPO="${RELP_REPO:-/home/rkathuria/relp-circuits}"
export RELP_ARTIFACTS="${RELP_ARTIFACTS:-/data/artifacts/rohan/relp-circuits}"

# Hugging Face caches all live on /data so models don't fill /home.
export HF_HOME="${RELP_ARTIFACTS}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# uv: project venv lives in the repo (default location). Only the wheel cache goes on
# /data, since it can grow into the GBs. Cache is on a different filesystem from the
# venv, so force copy mode — hardlinking across /data and /home silently leaves dist-info
# without the package's actual files (observed with nvidia-cudnn-cu12).
export UV_CACHE_DIR="${RELP_ARTIFACTS}/uv_cache"
export UV_LINK_MODE=copy

# Carry user's HF_TOKEN through if set in the shell that ran sbatch.
# sbatch propagates the submitting shell's env by default, so HF_TOKEN flows through.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[common.sh] WARNING: HF_TOKEN not set in env; downloads will be anonymous and may rate-limit." >&2
fi

# Print a one-line context header so logs are self-describing.
echo "[common.sh] host=$(hostname) job=${SLURM_JOB_ID:-none} repo=${RELP_REPO} artifacts=${RELP_ARTIFACTS}"
