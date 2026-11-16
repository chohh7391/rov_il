"""Every rov_lab task config must parse before Kit starts without loading ``pxr``.

Isaac Lab's scripts (Mimic annotate / generate, record_demos) import the task and parse its config before launching
Kit. If that loads the pip ``usd-core`` build of ``pxr``, Kit's own USD libraries then fail to load (undefined
``pxrInternal_v0_25_11`` symbols) and the app dies. Runs in a subprocess for a clean ``sys.modules``.

    PYTHONPATH=rov_lab/source <isaaclab-3.0-python> -m unittest discover -s rov_lab/tests -t rov_lab -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO: Path = Path(__file__).resolve().parents[3]
TASKS: tuple[str, ...] = (
    "ROVLab-SO101-PickStone-v0",
    "ROVLab-I90Bravo7-ValvePanel-v0",
    "ROVLab-I90Bravo7-ValvePanel-Ledge-v0",
    "ROVLab-I90Bravo7-ValvePanel-AbsPose-v0",
    "ROVLab-I90Bravo7-ValvePanel-Mimic-v0",
)
PROBE = """
import json, sys
import rov_lab.tasks.valve_panel  # the Mimic scripts' --external_callback module
from isaaclab_tasks.utils import parse_env_cfg
for task in sys.argv[1:]:
    parse_env_cfg(task, device="cpu", num_envs=2)
print("RESULT " + json.dumps({"pxr_loaded": "pxr" in sys.modules}))
"""


class TaskConfigsKitFree(unittest.TestCase):
    def test_parse_without_pxr(self) -> None:
        try:
            import isaaclab_tasks  # noqa: F401
        except ImportError:
            self.skipTest("isaaclab_tasks not installed in this interpreter")
        env = dict(os.environ, PYTHONPATH=str(REPO / "rov_lab" / "source") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        proc = subprocess.run([sys.executable, "-c", PROBE, *TASKS], capture_output=True, text=True, env=env, timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
        self.assertFalse(json.loads(line[len("RESULT "):])["pxr_loaded"], "a task config import loaded pxr")


if __name__ == "__main__":
    unittest.main()
