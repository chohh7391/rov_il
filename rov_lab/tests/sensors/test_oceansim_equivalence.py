"""Equivalence tests: rov_lab's Isaac Lab 3.0 underwater sensors vs OceanSim's own code.

OceanSim (``external_dependencies/OceanSim``) targets Isaac Sim 4.5/5.x and most of its sensor classes need a running
Kit app, so these tests use OceanSim's code where it can run standalone and its source text where it cannot:

* underwater camera: OceanSim's ``UW_render`` Warp kernel is imported from its file and run on CPU;
* DVL beam geometry: beam orientations are built with Isaac Sim's own ``euler_to_rot_matrix`` (the function OceanSim
  calls), loaded from the Isaac Sim 5.1 install with ``pxr`` stubbed;
* DVL noise transform: the matrix expression is read from OceanSim's ``DVLsensor.py`` with ``ast`` and evaluated;
* DVL velocity / dropout and barometer: OceanSim's formulas, guarded by checks that the source still contains them.

Run (no simulator needed):
    PYTHONPATH=rov_lab/source <isaaclab-3.0-python> -m unittest discover -s rov_lab/tests -v
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from rov_lab.assets.sensors import UWCameraCfg, uw_render
from rov_lab.assets.sensors.dvl import (
    DVLJanusPatternCfg,
    bottom_lock,
    dvl_body_velocity,
    dvl_janus_pattern,
    janus_noise_transform,
)
from rov_lab.assets.sensors import barometer_pressure_model

REPO = Path(__file__).resolve().parents[3]
OCEANSIM = REPO / "external_dependencies" / "OceanSim" / "isaacsim" / "oceansim"
ISAACSIM_ROTATIONS = (
    REPO / ".venv-rov_lab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.utils/isaacsim/core/utils/rotations.py"
)


def _load_module(name: str, path: Path, stub_modules: tuple[str, ...] = ()) -> types.ModuleType:
    class _AnyAttr:
        """Stand-in for ``pxr.Gf``: only used in annotations / helpers these tests do not call."""

        def __getattr__(self, name: str) -> type:
            return object

    for stub in stub_modules:
        if stub not in sys.modules:
            mod = types.ModuleType(stub)
            mod.Gf = _AnyAttr()
            sys.modules[stub] = mod
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _xyzw_from_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


@unittest.skipUnless(OCEANSIM.exists(), "OceanSim sources not found")
class UnderwaterCameraEquivalence(unittest.TestCase):
    """rov_lab ``uw_render`` (torch) vs OceanSim ``UW_render`` (Warp kernel) on identical RGB + depth."""

    @classmethod
    def setUpClass(cls) -> None:
        import warp as wp

        wp.init()
        cls.wp = wp
        cls.uw = _load_module("oceansim_uwrenderer_utils", OCEANSIM / "utils" / "UWrenderer_utils.py")

    def _oceansim(self, rgba: np.ndarray, depth: np.ndarray, cfg: UWCameraCfg) -> np.ndarray:
        wp = self.wp
        h, w = depth.shape
        raw = wp.array(rgba, dtype=wp.uint8, device="cpu")
        dep = wp.array(depth.astype(np.float32), dtype=wp.float32, device="cpu")
        out = wp.zeros((h, w, 4), dtype=wp.uint8, device="cpu")
        wp.launch(
            self.uw.UW_render,
            dim=(h, w),
            inputs=[raw, dep, wp.vec3f(*cfg.backscatter_value), wp.vec3f(*cfg.atten_coeff), wp.vec3f(*cfg.backscatter_coeff)],
            outputs=[out],
            device="cpu",
        )
        wp.synchronize()
        return out.numpy()[..., :3]

    def _compare(self, cfg: UWCameraCfg, depth: np.ndarray, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        rgba = rng.integers(0, 256, size=(*depth.shape, 4), dtype=np.uint8)
        ref = self._oceansim(rgba, depth, cfg).astype(np.int32)
        ours = uw_render(torch.from_numpy(rgba)[None], torch.from_numpy(depth)[None], cfg)[0].numpy().astype(np.int32)
        diff = np.abs(ref - ours)
        # identical math; float32 exp in Warp vs torch can move a value across an integer boundary (truncation)
        self.assertLessEqual(int(diff.max()), 1)
        self.assertGreaterEqual(float((diff == 0).mean()), 0.999)

    def test_default_water_matches(self) -> None:
        depth = np.random.default_rng(1).uniform(0.0, 60.0, size=(96, 128)).astype(np.float32)
        self._compare(UWCameraCfg(), depth)

    def test_turbid_water_matches(self) -> None:
        cfg = UWCameraCfg(backscatter_value=(0.05, 0.25, 0.35), atten_coeff=(0.4, 0.2, 0.1), backscatter_coeff=(0.3, 0.3, 0.6))
        depth = np.random.default_rng(2).uniform(0.0, 15.0, size=(64, 64)).astype(np.float32)
        self._compare(cfg, depth, seed=3)

    def test_default_parameters_match_oceansim(self) -> None:
        # OceanSim UW_Camera.initialize default UW_param: [backscatter_value, backscatter_coeff, atten_coeff]
        src = (OCEANSIM / "sensors" / "UW_Camera.py").read_text()
        self.assertIn("np.array([0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05 ])", src)
        cfg = UWCameraCfg()
        self.assertEqual(tuple(cfg.backscatter_value), (0.0, 0.31, 0.24))
        self.assertEqual(tuple(cfg.backscatter_coeff), (0.05, 0.05, 0.2))
        self.assertEqual(tuple(cfg.atten_coeff), (0.05, 0.05, 0.05))

    def test_open_water_matches(self) -> None:
        # no hit: OceanSim passes inf depth -> pure backscatter color
        depth = np.full((8, 8), np.inf, dtype=np.float32)
        self._compare(UWCameraCfg(), depth)


@unittest.skipUnless(OCEANSIM.exists(), "OceanSim sources not found")
class DVLEquivalence(unittest.TestCase):
    def setUp(self) -> None:
        self.src = (OCEANSIM / "sensors" / "DVLsensor.py").read_text()

    @unittest.skipUnless(ISAACSIM_ROTATIONS.exists(), "Isaac Sim 5.1 rotations.py not found")
    def test_beam_directions_match_oceansim(self) -> None:
        rot = _load_module("isaacsim_rotations_ref", ISAACSIM_ROTATIONS, stub_modules=("pxr",))
        self.assertIn("forward_axis=Gf.Vec3d(0, 0, -1)", self.src)
        for elevation, rotation in [(22.5, 45.0), (30.0, 0.0), (15.0, 30.0)]:
            # OceanSim attachDVL: per-beam euler (deg), extrinsic, applied to forward axis -Z
            eulers = [[elevation, 0.0, rotation], [0.0, elevation, rotation], [-elevation, 0.0, rotation], [0.0, -elevation, rotation]]
            ref = np.stack([rot.euler_to_rot_matrix(np.array(e), degrees=True) @ np.array([0.0, 0.0, -1.0]) for e in eulers])
            _, ours = dvl_janus_pattern(DVLJanusPatternCfg(elevation=elevation, rotation=rotation), "cpu")
            np.testing.assert_allclose(ours.numpy(), ref, atol=1e-6)

    def test_noise_transform_matches_oceansim_source(self) -> None:
        tree = ast.parse(self.src)
        expr = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(ast.unparse(t) == "self._transform" for t in node.targets):
                expr = ast.unparse(node.value)
        self.assertIsNotNone(expr, "OceanSim DVL transform not found")
        for elevation in (22.5, 30.0):
            ref = eval(expr, {"np": np, "sinElev": math.sin(math.radians(elevation)), "cosElev": math.cos(math.radians(elevation))})  # noqa: S307
            np.testing.assert_allclose(janus_noise_transform(elevation).numpy(), ref, rtol=1e-6)

    @unittest.skipUnless(ISAACSIM_ROTATIONS.exists(), "Isaac Sim 5.1 rotations.py not found")
    def test_velocity_matches_oceansim(self) -> None:
        # OceanSim get_linear_vel: vel = R(world_orient).T @ world_vel (+ T @ beam_noise); world_orient is wxyz
        self.assertIn("vel = rot_m.T @ world_vel", self.src)
        self.assertIn("vel[j] += self._transform[j][i] * sample[i]", self.src)
        rot = _load_module("isaacsim_rotations_ref2", ISAACSIM_ROTATIONS, stub_modules=("pxr",))
        rng = np.random.default_rng(4)
        T = janus_noise_transform(22.5).numpy()
        for _ in range(50):
            q_wxyz = rng.normal(size=4)
            q_wxyz /= np.linalg.norm(q_wxyz)
            v_w = rng.normal(size=3)
            noise = rng.normal(scale=0.01, size=4)
            ref = rot.quat_to_rot_matrix(q_wxyz).T @ v_w + T @ noise
            ours = dvl_body_velocity(
                torch.tensor(v_w, dtype=torch.float32)[None], torch.tensor(_xyzw_from_wxyz(q_wxyz), dtype=torch.float32)[None]
            )[0].numpy() + T @ noise
            np.testing.assert_allclose(ours, ref, atol=1e-5)

    def test_dropout_rule_matches_oceansim(self) -> None:
        self.assertIn("if if_hit.count(False) >= self._num_beams_out_range_threshold:", self.src)
        patterns = torch.tensor([[(m >> b) & 1 for b in range(4)] for m in range(16)], dtype=torch.bool)
        for threshold in (1, 2, 3):
            ours = bottom_lock(patterns, threshold)[:, 0]
            ref = torch.tensor([list(p).count(False) < threshold for p in patterns.tolist()])
            self.assertTrue(torch.equal(ours, ref))


@unittest.skipUnless(OCEANSIM.exists(), "OceanSim sources not found")
class BarometerEquivalence(unittest.TestCase):
    def test_pressure_matches_oceansim(self) -> None:
        src = (OCEANSIM / "sensors" / "BarometerSensor.py").read_text()
        self.assertIn("pressure = self._atmosphere_pressure + self._water_density * self._g * depth", src)
        surface, rho, g, p_atm = 3.43389, 1000.0, 9.81, 101325.0
        z = torch.linspace(-10.0, 6.0, 33, dtype=torch.float64)
        ours = barometer_pressure_model(z, surface, rho, g, p_atm)
        # OceanSim get_pressure: depth = surface - z if z < surface else 0
        ref = torch.tensor([p_atm + rho * g * (surface - float(zi) if float(zi) < surface else 0.0) for zi in z], dtype=torch.float64)
        torch.testing.assert_close(ours, ref, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
