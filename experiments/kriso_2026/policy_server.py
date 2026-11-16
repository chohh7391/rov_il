"""Serve a trained LeRobot policy to the Isaac Sim evaluation process over a local socket.

Runs in the `fire` conda env (lerobot 0.4.4) so the simulator env does not need LeRobot.
usage: python policy_server.py --ckpt checkpoints/act_kriso/checkpoints/last/pretrained_model --port 50555
"""

import argparse
from multiprocessing.connection import Listener

import numpy as np
import torch
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.control_utils import predict_action

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--port", type=int, default=50555)
# inference-time overrides (defaults keep the trained config)
ap.add_argument("--temporal_ensemble", type=float, default=None, help="ACT temporal ensembling coeff (forces n_action_steps=1)")
ap.add_argument("--n_action_steps", type=int, default=None)
args = ap.parse_args()

cfg = PreTrainedConfig.from_pretrained(args.ckpt)
if args.temporal_ensemble is not None:
    cfg.temporal_ensemble_coeff = args.temporal_ensemble
    cfg.n_action_steps = 1
elif args.n_action_steps is not None:
    cfg.n_action_steps = args.n_action_steps
policy = get_policy_class(cfg.type).from_pretrained(args.ckpt, config=cfg)
policy.to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt,
                                     preprocessor_overrides={"device_processor": {"device": "cuda"}})
device = torch.device("cuda")
print(f"[server] loaded {cfg.type} from {args.ckpt} (n_action_steps={cfg.n_action_steps}, "
      f"temporal_ensemble={getattr(cfg, 'temporal_ensemble_coeff', None)})", flush=True)

with Listener(("127.0.0.1", args.port), authkey=b"kriso") as listener:
    while True:
        conn = listener.accept()
        print("[server] client connected", flush=True)
        try:
            while True:
                msg = conn.recv()
                if msg["cmd"] == "reset":
                    policy.reset()
                    conn.send({"ok": True})
                elif msg["cmd"] == "act":
                    # arrays arrive as (bytes, dtype, shape) to stay independent of numpy pickle versions
                    obs = {k: np.frombuffer(b, dtype=dt).reshape(sh).copy() for k, (b, dt, sh) in msg["obs"].items()}
                    action = predict_action(obs, policy, device, pre, post, use_amp=False,
                                            task=msg.get("task"), robot_type="bluerov_so101")
                    a = action.cpu().numpy().astype(np.float32).reshape(-1)
                    conn.send({"action": (a.tobytes(), "float32", tuple(a.shape))})
                elif msg["cmd"] == "close":
                    conn.send({"ok": True})
                    break
        except EOFError:
            pass
        conn.close()
        print("[server] client disconnected", flush=True)
