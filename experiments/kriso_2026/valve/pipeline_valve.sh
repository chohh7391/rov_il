#!/usr/bin/env bash
# Valve-panel IL pipeline after collection: wait for collect_valve.py -> LeRobot conversion -> ACT training -> eval.
# usage: pipeline_valve.sh [steps]   (run from anywhere; logs go to runs/kriso_2026/)
set -u
STEPS=${1:-30000}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
FIRE=${FIRE_BIN:-$HOME/anaconda3/envs/fire/bin}
LOG=runs/kriso_2026/pipeline_valve.log
DEMOS=datasets/kriso_2026/valve_demos
DS=datasets/kriso_2026/lerobot_v3/kriso_valve
CK=checkpoints/kriso_2026/act_valve
echo "[pipeline] start $(date)" >> $LOG
until grep -q "\[collect\] DONE" runs/kriso_2026/collect_valve.log 2>/dev/null; do
  pgrep -f "[c]ollect_valve.py" > /dev/null || { echo "[pipeline] collector not running and not done" >> $LOG; exit 1; }
  sleep 60
done
echo "[pipeline] collection done $(date)" >> $LOG
$FIRE/python experiments/kriso_2026/valve/to_lerobot_valve.py --src $DEMOS --root $DS --fps 30 \
  > runs/kriso_2026/to_lerobot_valve.log 2>&1 || { echo "[pipeline] conversion failed" >> $LOG; exit 1; }
tail -n 1 runs/kriso_2026/to_lerobot_valve.log >> $LOG
rm -rf $CK
$FIRE/lerobot-train \
  --dataset.repo_id=rov_il/kriso_valve --dataset.root=$DS \
  --policy.type=act --policy.push_to_hub=false --policy.device=cuda --policy.chunk_size=20 --policy.n_action_steps=10 \
  --output_dir=$CK --job_name=act_valve --wandb.enable=false \
  --batch_size=16 --steps=$STEPS --save_freq=10000 --log_freq=500 --num_workers=8 \
  > runs/kriso_2026/train_act_valve.log 2>&1 || { echo "[pipeline] training failed" >> $LOG; exit 1; }
echo "[pipeline] training done $(date)" >> $LOG
CKPT=$CK/checkpoints/$(printf "%06d" $STEPS)/pretrained_model
bash experiments/kriso_2026/valve/run_eval_valve.sh $CKPT runs/kriso_2026/eval_act_valve --episodes 60 >> $LOG 2>&1
echo "[pipeline] PIPELINE_DONE $(date)" >> $LOG
