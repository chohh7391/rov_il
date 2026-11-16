#!/usr/bin/env bash
# usage: run_eval_valve.sh <ckpt pretrained_model dir> <out dir> [extra eval_valve.py args...]
# Starts the LeRobot policy server (fire env), runs eval_valve.py (Isaac Lab 3.0 env), then stops the server.
set -u
CKPT=$1; OUT=$2; shift 2
PORT=${PORT:-50555}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
FIRE_PY=${FIRE_PY:-$HOME/anaconda3/envs/fire/bin/python}
ISAAC_PY=${ISAAC_PY:-$HOME/IsaacLabs/3.0.0/env_isaaclab/bin/python}
mkdir -p "$OUT"
$FIRE_PY $ROOT/experiments/kriso_2026/policy_server.py --ckpt "$CKPT" --port $PORT ${SERVER_ARGS:-} > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 120); do grep -q "\[server\] loaded" "$OUT/server.log" && break; kill -0 $SRV 2>/dev/null || { echo "server died"; tail -n 20 "$OUT/server.log"; exit 1; }; sleep 1; done
cd $ROOT && PYTHONPATH=rov_lab/source:experiments/kriso_2026/valve OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PY \
    experiments/kriso_2026/valve/eval_valve.py --port $PORT --out "$OUT" "$@" > "$OUT/eval.log" 2>&1
echo "EVAL_EXIT $?"
grep "\[eval\] SUMMARY" "$OUT/eval.log"
