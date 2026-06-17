#!/usr/bin/env bash
set -euo pipefail

SSH_PORT="${SSH_PORT:-22}"
REMOTE_HOST="${REMOTE_HOST:-xiechunyang@112.74.164.141}"
REMOTE_BFM_ROOT="${REMOTE_BFM_ROOT:-/home/xiechunyang/wt_ws/wt_wbc/BFM-Zero-ManagerOnly}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-/home/thl/wt_wbc/BFM-Zero-ManagerOnly/results}"
LOCAL_REPO_ROOT="${LOCAL_REPO_ROOT:-/home/thl/wt_wbc/BFM-Zero-ManagerOnly}"

usage() {
  cat <<'EOF'
Usage:
  ./wt_download_result.sh <result_name> [--with-assets] [--dry-run]

Examples:
  ./wt_download_result.sh bfmzero-manager-nohead-minimal-cuda0
  ./wt_download_result.sh bfmzero-manager-nohead-minimal-cuda0 --with-assets

Environment overrides:
  REMOTE_HOST        default: xiechunyang@112.74.164.141
  SSH_PORT           default: 22
  REMOTE_BFM_ROOT    default: /home/xiechunyang/wt_ws/wt_wbc/BFM-Zero-ManagerOnly
  LOCAL_RESULTS_ROOT default: /home/thl/wt_wbc/BFM-Zero/results
  LOCAL_REPO_ROOT    default: /home/thl/wt_wbc/BFM-Zero-ManagerOnly

This downloads the minimal files needed for tracking inference:
  config.json
  checkpoint/model/config.json
  checkpoint/model/init_kwargs.json or init_kwargs.pkl
  checkpoint/model/model.safetensors
  small checkpoint metadata, if present
  exported/FBcprAuxModel.onnx, if present

It intentionally does not download checkpoint/buffers or training logs.
EOF
}

result_name=""
with_assets=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --with-assets)
      with_assets=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$result_name" ]]; then
        echo "Only one result_name is expected, got extra argument: $1" >&2
        usage >&2
        exit 2
      fi
      result_name="$1"
      shift
      ;;
  esac
done

if [[ -z "$result_name" ]]; then
  usage >&2
  exit 2
fi

if [[ "$result_name" == */* ]]; then
  echo "Please pass only the results directory name, not a path: $result_name" >&2
  exit 2
fi

rsync_dry_run=()
if [[ "$dry_run" -eq 1 ]]; then
  rsync_dry_run+=(--dry-run)
fi

remote_result="${REMOTE_BFM_ROOT%/}/results/${result_name}/"
local_result="${LOCAL_RESULTS_ROOT%/}/${result_name}/"

mkdir -p "$local_result"

echo "Downloading tracking checkpoint:"
echo "  remote: ${REMOTE_HOST}:${remote_result}"
echo "  local:  ${local_result}"

rsync -avz --prune-empty-dirs "${rsync_dry_run[@]}" \
  -e "ssh -p ${SSH_PORT}" \
  --include='*/' \
  --include='/config.json' \
  --include='/manager_env_info.json' \
  --include='/checkpoint/config.json' \
  --include='/checkpoint/train_status.json' \
  --include='/checkpoint/model/config.json' \
  --include='/checkpoint/model/init_kwargs.json' \
  --include='/checkpoint/model/init_kwargs.pkl' \
  --include='/checkpoint/model/model.safetensors' \
  --include='/exported/FBcprAuxModel.onnx' \
  --exclude='*' \
  "${REMOTE_HOST}:${remote_result}" \
  "$local_result"

if [[ "$with_assets" -eq 1 ]]; then
  local_assets="${LOCAL_REPO_ROOT%/}/bfm/data/robots/g1/"
  mkdir -p "$local_assets"
  echo "Downloading robot assets:"
  echo "  remote: ${REMOTE_HOST}:${REMOTE_BFM_ROOT%/}/bfm/data/robots/g1/"
  echo "  local:  ${local_assets}"
  rsync -avz "${rsync_dry_run[@]}" \
    -e "ssh -p ${SSH_PORT}" \
    "${REMOTE_HOST}:${REMOTE_BFM_ROOT%/}/bfm/data/robots/g1/" \
    "$local_assets"
fi

cat <<EOF

Done.

Use it with:
  --model-folder ${local_result%/}

Motion data is not downloaded by this script. Pass --data-path to your local pkl/npz data when running inference.
EOF
