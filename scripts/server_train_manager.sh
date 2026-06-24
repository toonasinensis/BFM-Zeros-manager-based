#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE_USER="${BFM_REMOTE_USER:-xiechunyang}"
REMOTE_HOST="${BFM_REMOTE_HOST:-47.107.87.125}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_DIR="${BFM_REMOTE_DIR:-/home/xiechunyang/wt_ws/wt_wbc/BFM-Zero-ManagerOnly}"

MOTION_FILE="${BFM_MOTION_FILE:-${REMOTE_DIR}/bfm/data/named_lafan_10s}"
#下面这些非必要不改：：：：：：：：：：：：：
ONLINE_ENVS="${BFM_ONLINE_ENVS:-1024}"
NUM_ENV_STEPS="${BFM_NUM_ENV_STEPS:-384000000}"
NUM_SEED_STEPS="${BFM_NUM_SEED_STEPS:-10240}"
UPDATE_EVERY="${BFM_UPDATE_EVERY:-1024}"
NUM_AGENT_UPDATES="${BFM_NUM_AGENT_UPDATES:-16}"
BATCH_SIZE="${BFM_BATCH_SIZE:-1024}"
BUFFER_SIZE="${BFM_BUFFER_SIZE:-3000000}"
CHECKPOINT_EVERY_STEPS="${BFM_CHECKPOINT_EVERY_STEPS:-1024000}"
LOG_EVERY_UPDATES="${BFM_LOG_EVERY_UPDATES:-10240}"

USE_WANDB="${BFM_USE_WANDB:-1}"
WANDB_ENTITY="${BFM_WANDB_ENTITY:-xiechunyang1-hajimi}"
WANDB_PROJECT="${BFM_WANDB_PROJECT:-hajimi}"

#上面面这些非必要不改：：：：：：：：：：：：：

#TODO: TEST ANGLE SCALE的影响
BASE_ANG_VEL_OBS_SCALE="${BFM_BASE_ANG_VEL_OBS_SCALE:-0.25}"
EXPERT_BASE_ANG_VEL_OBS_SCALE="${BFM_EXPERT_BASE_ANG_VEL_OBS_SCALE:-0.25}"

#下面一定要改：：：：：：：：：：：：：：：：：：：：：：：
GPU="${BFM_CUDA_VISIBLE_DEVICES:-2}"
WANDB_GROUP="${BFM_WANDB_GROUP:-angle_scale_compare}"
RUN_NAME="${BFM_RUN_NAME:-obs0.25expert0.25-$(date +%Y%m%d-%H%M%S)}"
#上面一定要改：：：：：：：：：：：：：：：：：：：：：：：

if [[ -n "$RUN_NAME" ]]; then
  DEFAULT_SESSION="$RUN_NAME"
  DEFAULT_WORK_DIR="results/$RUN_NAME"
  DEFAULT_WANDB_NAME="$RUN_NAME"
else
  DEFAULT_SESSION="傻逼你没改"
  DEFAULT_WORK_DIR="results/傻逼你没改"
  DEFAULT_WANDB_NAME="傻逼你没改-$(date +%Y%m%d-%H%M%S)"
fi
SESSION="${BFM_TMUX_SESSION:-$DEFAULT_SESSION}"
WORK_DIR="${BFM_WORK_DIR:-$DEFAULT_WORK_DIR}"
WANDB_NAME="${BFM_WANDB_NAME:-$DEFAULT_WANDB_NAME}"

ENABLE_EVAL="${BFM_ENABLE_EVAL:-1}"
PRIORITIZATION="${BFM_PRIORITIZATION:-1}"
ENABLE_DOMAIN_RANDOMIZATION="${BFM_ENABLE_DOMAIN_RANDOMIZATION:-1}"
CHECKPOINT_BUFFER="${BFM_CHECKPOINT_BUFFER:-1}"
DISABLE_TQDM="${BFM_DISABLE_TQDM:-1}"
TRAINING_MAX_NUM_SEQS="${BFM_TRAINING_MAX_NUM_SEQS:-}"

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)
SSH_CMD=(ssh "${SSH_OPTS[@]}")
RSYNC_SSH="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

usage() {
  cat <<USAGE
Usage: $(basename "$0") <command>

Commands:
  probe          Check SSH, remote directory, tmux, uv, GPU, and wandb login hints.
  sync           Upload this repo's code plus required robot assets and named_lafan_10s data.
  tools          Ensure uv/uvx are available on the remote under ~/.local/bin.
  sync-auth      Copy local ~/.netrc to the remote for wandb login.
  setup          Create/update the remote .venv with uv and run a compile check.
  smoke          Run a tiny no-wandb Isaac training smoke on the remote GPU.
  start          Start long wandb training in a detached remote tmux session.
  deploy-start   Run sync, setup, then start.
  status         Show tmux/process/GPU status and the latest training log lines.
  logs           Follow the remote training stdout log.
  stop           Send Ctrl-C to the remote tmux training session.

Useful environment overrides:
  BFM_RUN_NAME=$RUN_NAME       # Sets default tmux session, work dir, and wandb name.
  BFM_REMOTE_USER=$REMOTE_USER
  BFM_REMOTE_HOST=$REMOTE_HOST
  BFM_REMOTE_DIR=$REMOTE_DIR
  BFM_CUDA_VISIBLE_DEVICES=$GPU
  BFM_TMUX_SESSION=$SESSION
  BFM_WORK_DIR=$WORK_DIR
  BFM_NUM_ENV_STEPS=$NUM_ENV_STEPS
  BFM_ONLINE_ENVS=$ONLINE_ENVS
  BFM_ENABLE_EVAL=$ENABLE_EVAL
  BFM_PRIORITIZATION=$PRIORITIZATION
  BFM_ENABLE_DOMAIN_RANDOMIZATION=$ENABLE_DOMAIN_RANDOMIZATION
  BFM_CHECKPOINT_BUFFER=$CHECKPOINT_BUFFER
  BFM_BASE_ANG_VEL_OBS_SCALE=$BASE_ANG_VEL_OBS_SCALE
  BFM_EXPERT_BASE_ANG_VEL_OBS_SCALE=$EXPERT_BASE_ANG_VEL_OBS_SCALE
  BFM_WANDB_NAME=$WANDB_NAME

This script intentionally does not store passwords. Use SSH keys, or let ssh/rsync prompt.
USAGE
}

remote_bash() {
  local remote_dir_q
  remote_dir_q="$(printf "%q" "$REMOTE_DIR")"
  "${SSH_CMD[@]}" "$REMOTE" "REMOTE_DIR=${remote_dir_q} bash -s"
}

remote_quote() {
  printf "%q" "$1"
}

bool_flag() {
  local enabled="$1"
  local flag="$2"
  local no_flag="$3"
  if [[ "$enabled" == "1" || "$enabled" == "true" || "$enabled" == "yes" || "$enabled" == "on" ]]; then
    printf "%s" "$flag"
  else
    printf "%s" "$no_flag"
  fi
}

probe() {
  remote_bash <<'REMOTE'
set -euo pipefail
echo "remote=$(hostname)"
echo "pwd=$(pwd)"
echo "target=${REMOTE_DIR}"
if [[ -d "${REMOTE_DIR}" ]]; then
  ls -ld "${REMOTE_DIR}"
  find "${REMOTE_DIR}/bfm/data/named_lafan_10s" -maxdepth 1 -type f 2>/dev/null | wc -l | awk '{print "named_lafan_10s_files="$1}'
fi
echo "tmux=$(command -v tmux || true)"
echo "uv=$(command -v uv || command -v "${HOME}/.local/bin/uv" || true)"
echo "python3=$(command -v python3 || true)"
if [[ -f "${HOME}/.netrc" ]]; then
  echo "wandb_netrc=present"
else
  echo "wandb_netrc=missing"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
REMOTE
}

sync_repo() {
  cd "$REPO_ROOT"
  "${SSH_CMD[@]}" "$REMOTE" "mkdir -p $(remote_quote "$REMOTE_DIR")/bfm/data"

  rsync -az --delete \
    --filter='P /data/***' \
    --exclude='/data/***' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -e "$RSYNC_SSH" \
    bfm/ "$REMOTE:$REMOTE_DIR/bfm/"

  rsync -az --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -e "$RSYNC_SSH" \
    scripts/ "$REMOTE:$REMOTE_DIR/scripts/"

  rsync -az --delete \
    -e "$RSYNC_SSH" \
    docs/ "$REMOTE:$REMOTE_DIR/docs/"

  rsync -az \
    -e "$RSYNC_SSH" \
    README.md README_MANAGER_ONLY.md pyproject.toml uv.lock "$REMOTE:$REMOTE_DIR/"

  rsync -az \
    -e "$RSYNC_SSH" \
    bfm/data/name.py "$REMOTE:$REMOTE_DIR/bfm/data/"

  rsync -az --delete \
    -e "$RSYNC_SSH" \
    bfm/data/robots/ "$REMOTE:$REMOTE_DIR/bfm/data/robots/"

  rsync -az --delete \
    -e "$RSYNC_SSH" \
    bfm/data/named_lafan_10s/ "$REMOTE:$REMOTE_DIR/bfm/data/named_lafan_10s/"

  remote_bash <<'REMOTE'
set -euo pipefail
cd "${REMOTE_DIR}"
find bfm -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find scripts -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find bfm scripts -type d -empty -delete 2>/dev/null || true
echo "sync_done target=${REMOTE_DIR}"
find bfm/data/named_lafan_10s -maxdepth 1 -type f | wc -l | awk '{print "named_lafan_10s_files="$1}'
REMOTE
}

ensure_remote_uv() {
  cd "$REPO_ROOT"
  if "${SSH_CMD[@]}" "$REMOTE" 'command -v uv >/dev/null 2>&1 || command -v "$HOME/.local/bin/uv" >/dev/null 2>&1'; then
    return
  fi

  local local_uv local_uvx
  local_uv="$(command -v uv || true)"
  local_uvx="$(command -v uvx || true)"
  if [[ -z "$local_uv" ]]; then
    echo "uv is missing locally and remotely; install uv locally or on the remote first." >&2
    exit 2
  fi

  "${SSH_CMD[@]}" "$REMOTE" 'mkdir -p "$HOME/.local/bin"'
  rsync -az -e "$RSYNC_SSH" "$local_uv" "$REMOTE:/home/$REMOTE_USER/.local/bin/uv"
  if [[ -n "$local_uvx" ]]; then
    rsync -az -e "$RSYNC_SSH" "$local_uvx" "$REMOTE:/home/$REMOTE_USER/.local/bin/uvx"
  fi
  "${SSH_CMD[@]}" "$REMOTE" 'chmod +x "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx" 2>/dev/null || true; "$HOME/.local/bin/uv" --version'
}

sync_auth() {
  if [[ ! -f "$HOME/.netrc" ]]; then
    echo "Local ~/.netrc is missing; run 'wandb login' locally or log in on the remote." >&2
    exit 2
  fi
  "${SSH_CMD[@]}" "$REMOTE" 'mkdir -p "$HOME/.config/wandb"'
  rsync -az -e "$RSYNC_SSH" "$HOME/.netrc" "$REMOTE:/home/$REMOTE_USER/.netrc"
  "${SSH_CMD[@]}" "$REMOTE" 'chmod 600 "$HOME/.netrc"; test -f "$HOME/.netrc" && echo "wandb_netrc=present"'
}

setup_remote() {
  ensure_remote_uv
  remote_bash <<'REMOTE'
set -euo pipefail
cd "${REMOTE_DIR}"
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is missing. Install uv on the remote or put it in ~/.local/bin." >&2
  exit 2
fi
if [[ "${BFM_SERVER_KEEP_HUMENV:-0}" != "1" ]]; then
  cp pyproject.toml pyproject.toml.bfm-server.bak
  restore_pyproject() {
    if [[ -f pyproject.toml.bfm-server.bak ]]; then
      mv pyproject.toml.bfm-server.bak pyproject.toml
    fi
  }
  trap restore_pyproject EXIT
  python3 - <<'PY'
from pathlib import Path

path = Path("pyproject.toml")
text = path.read_text()
text = text.replace('    "humenv",\n', "")
text = text.replace('humenv = { git = "https://github.com/facebookresearch/humenv.git" }\n', "")
path.write_text(text)
PY
fi
uv sync
if declare -f restore_pyproject >/dev/null 2>&1; then
  restore_pyproject
  trap - EXIT
fi
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONBREAKPOINT=0
export BFM_DISABLE_TORCH_COMPILE=1
.venv/bin/python -m compileall -q bfm scripts
.venv/bin/python - <<'PY'
import importlib
for name in ["torch", "wandb", "tyro", "numpy"]:
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'ok')}")
PY
echo "setup_done"
REMOTE
}

smoke_remote() {
  local smoke_dir="${BFM_SMOKE_WORK_DIR:-results/server-smoke-named-lafan-scale1}"
  remote_bash <<REMOTE
set -euo pipefail
cd "${REMOTE_DIR}"
export PATH="\${HOME}/.local/bin:\${PATH}"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONBREAKPOINT=0
export BFM_DISABLE_TORCH_COMPILE=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE=disabled
.venv/bin/python -m bfm.train_manager_standalone \\
  --settings.device cuda:0 \\
  --settings.agent-device cuda \\
  --settings.buffer-device cuda \\
  --settings.work-dir "${smoke_dir}" \\
  --settings.no-compile-model \\
  --settings.motion-file "${MOTION_FILE}" \\
  --settings.base-ang-vel-obs-scale "${BASE_ANG_VEL_OBS_SCALE}" \\
  --settings.expert-base-ang-vel-obs-scale "${EXPERT_BASE_ANG_VEL_OBS_SCALE}" \\
  --settings.online-parallel-envs 4 \\
  --settings.num-env-steps 128 \\
  --settings.num-seed-steps 32 \\
  --settings.update-agent-every 32 \\
  --settings.num-agent-updates 1 \\
  --settings.batch-size 32 \\
  --settings.buffer-size 4096 \\
  --settings.checkpoint-every-steps 1000000000 \\
  --settings.no-checkpoint-buffer \\
  --settings.no-enable-eval \\
  --settings.no-prioritization \\
  --settings.no-enable-domain-randomization \\
  --settings.no-use-wandb \\
  --settings.disable-tqdm \\
  --settings.training-max-num-seqs 4
REMOTE
}

start_remote() {
  local eval_flag prioritization_flag dr_flag checkpoint_buffer_flag wandb_flag tqdm_flag training_max_line
  eval_flag="$(bool_flag "$ENABLE_EVAL" "--settings.enable-eval" "--settings.no-enable-eval")"
  prioritization_flag="$(bool_flag "$PRIORITIZATION" "--settings.prioritization" "--settings.no-prioritization")"
  dr_flag="$(bool_flag "$ENABLE_DOMAIN_RANDOMIZATION" "--settings.enable-domain-randomization" "--settings.no-enable-domain-randomization")"
  checkpoint_buffer_flag="$(bool_flag "$CHECKPOINT_BUFFER" "--settings.checkpoint-buffer" "--settings.no-checkpoint-buffer")"
  wandb_flag="$(bool_flag "$USE_WANDB" "--settings.use-wandb" "--settings.no-use-wandb")"
  tqdm_flag="$(bool_flag "$DISABLE_TQDM" "--settings.disable-tqdm" "--settings.no-disable-tqdm")"
  training_max_line=""
  if [[ -n "$TRAINING_MAX_NUM_SEQS" ]]; then
    training_max_line=" \\
  --settings.training-max-num-seqs \"${TRAINING_MAX_NUM_SEQS}\""
  fi

  remote_bash <<REMOTE
set -euo pipefail
cd "${REMOTE_DIR}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  tmux ls
  exit 3
fi
mkdir -p "${WORK_DIR}"
cat > "${WORK_DIR}/run_train.sh" <<'RUNSCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd "${REMOTE_DIR}"
export PATH="\${HOME}/.local/bin:\${PATH}"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONBREAKPOINT=0
export BFM_DISABLE_TORCH_COMPILE=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_DIR="\${WANDB_DIR:-${REMOTE_DIR}/_wandb}"
mkdir -p "\${WANDB_DIR}" "${WORK_DIR}"
LOG_FILE="${WORK_DIR}/train_stdout.log"
exec > >(tee -a "\${LOG_FILE}") 2>&1
echo "\$\$" > "${WORK_DIR}/train.pid"
echo "started_at=\$(date --iso-8601=seconds)"
echo "host=\$(hostname)"
echo "cuda_visible_devices=\${CUDA_VISIBLE_DEVICES}"
echo "work_dir=${WORK_DIR}"
echo "motion_file=${MOTION_FILE}"
echo "base_ang_vel_obs_scale=${BASE_ANG_VEL_OBS_SCALE}"
echo "expert_base_ang_vel_obs_scale=${EXPERT_BASE_ANG_VEL_OBS_SCALE}"
echo "wandb_name=${WANDB_NAME}"
set +e
.venv/bin/python -m bfm.train_manager_standalone \\
  --settings.device cuda:0 \\
  --settings.agent-device cuda \\
  --settings.buffer-device cuda \\
  --settings.work-dir "${WORK_DIR}" \\
  --settings.no-compile-model \\
  --settings.motion-file "${MOTION_FILE}" \\
  --settings.base-ang-vel-obs-scale "${BASE_ANG_VEL_OBS_SCALE}" \\
  --settings.expert-base-ang-vel-obs-scale "${EXPERT_BASE_ANG_VEL_OBS_SCALE}" \\
  --settings.online-parallel-envs "${ONLINE_ENVS}" \\
  --settings.num-env-steps "${NUM_ENV_STEPS}" \\
  --settings.log-every-updates "${LOG_EVERY_UPDATES}" \\
  --settings.update-agent-every "${UPDATE_EVERY}" \\
  --settings.num-seed-steps "${NUM_SEED_STEPS}" \\
  --settings.num-agent-updates "${NUM_AGENT_UPDATES}" \\
  --settings.batch-size "${BATCH_SIZE}" \\
  --settings.buffer-size "${BUFFER_SIZE}" \\
  --settings.checkpoint-every-steps "${CHECKPOINT_EVERY_STEPS}" \\
  ${checkpoint_buffer_flag} \\
  ${eval_flag} \\
  --settings.eval-every-steps "${CHECKPOINT_EVERY_STEPS}" \\
  ${prioritization_flag} \\
  ${dr_flag} \\
  ${wandb_flag} \\
  ${tqdm_flag} \\
  --settings.wandb-name "${WANDB_NAME}" \\
  --settings.wandb-entity "${WANDB_ENTITY}" \\
  --settings.wandb-project "${WANDB_PROJECT}" \\
  --settings.wandb-group "${WANDB_GROUP}"${training_max_line}
status=\$?
echo "\${status}" > "${WORK_DIR}/train.exit"
echo "finished_at=\$(date --iso-8601=seconds) status=\${status}"
exit "\${status}"
RUNSCRIPT
chmod +x "${WORK_DIR}/run_train.sh"
tmux new-session -d -s "${SESSION}" "bash ${REMOTE_DIR}/${WORK_DIR}/run_train.sh"
echo "started tmux session=${SESSION}"
echo "work_dir=${WORK_DIR}"
echo "log=${WORK_DIR}/train_stdout.log"
REMOTE
}

status_remote() {
  remote_bash <<REMOTE
set -euo pipefail
cd "${REMOTE_DIR}"
echo "tmux:"
tmux ls 2>/dev/null || true
echo
echo "pid:"
if [[ -f "${WORK_DIR}/train.pid" ]]; then
  pid="\$(cat "${WORK_DIR}/train.pid")"
  ps -p "\${pid}" -o pid,ppid,stat,etime,cmd 2>/dev/null || true
else
  echo "missing ${WORK_DIR}/train.pid"
fi
if [[ -f "${WORK_DIR}/train.exit" ]]; then
  printf "train.exit="
  cat "${WORK_DIR}/train.exit"
fi
echo
echo "gpu:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
echo
echo "log tail:"
tail -n 60 "${WORK_DIR}/train_stdout.log" 2>/dev/null || true
REMOTE
}

logs_remote() {
  "${SSH_CMD[@]}" -t "$REMOTE" "cd $(remote_quote "$REMOTE_DIR") && tail -f $(remote_quote "$WORK_DIR/train_stdout.log")"
}

stop_remote() {
  remote_bash <<REMOTE
set -euo pipefail
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux send-keys -t "${SESSION}" C-c
  echo "sent Ctrl-C to tmux session ${SESSION}"
else
  echo "no tmux session named ${SESSION}"
fi
REMOTE
}

cmd="${1:-}"
case "$cmd" in
  probe) probe ;;
  sync) sync_repo ;;
  tools) ensure_remote_uv ;;
  sync-auth) sync_auth ;;
  setup) setup_remote ;;
  smoke) smoke_remote ;;
  start) start_remote ;;
  deploy-start) sync_repo; setup_remote; sync_auth; start_remote ;;
  status) status_remote ;;
  logs) logs_remote ;;
  stop) stop_remote ;;
  ""|-h|--help|help) usage ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
