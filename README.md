cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

MUJOCO_GL=glfw PYTHONBREAKPOINT=0 /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.tracking_inference_mujoco \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --data-path /home/thl/wt_wbc/BFM-Zero-ManagerOnly/bfm/data/lafan_29dof.pkl \
  --motion-list 25 \
  --steps 100000 \
  --device cpu \
  --policy-runtime onnx \
  --no-headless \
  --show-reference \
  --real-time \
  --progress-every 200