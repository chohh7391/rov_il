"""Known-answer tests for rov_lab's underwater sensor models (independent of OceanSim).

Each test sets up a case whose answer can be derived by hand (flat seabed at a known altitude, pure rotation with a
known lever arm, zero/infinite viewing distance, known depth) and checks the model output against it.

Run (no simulator needed):
    PYTHONPATH=rov_lab/source <isaaclab-3.0-python> -m unittest discover -s rov_lab/tests -v
"""

from __future__ import annotations

import math
import unittest

import torch

from rov_lab.assets.sensors import UWCameraCfg, uw_render
from rov_lab.assets.sensors.dvl import DVLJanusPatternCfg, dvl_body_velocity, dvl_janus_pattern, janus_noise_transform
from rov_lab.assets.sensors import barometer_pressure_model
from rov_lab.assets.sensors.dvl import dvl_ranges_from_hits, dvl_ray_origins_w


def quat_xyzw_from_axis_angle(axis: tuple[float, float, float], angle: float) -> torch.Tensor:
    a = torch.tensor(axis, dtype=torch.float32)
    a = a / a.norm()
    return torch.cat([a * math.sin(angle / 2), torch.tensor([math.cos(angle / 2)])])[None]


class UnderwaterImageModel(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = UWCameraCfg()
        self.rgb = torch.tensor([[[[200, 120, 40]]]], dtype=torch.uint8)  # (1, 1, 1, 3)

    def test_zero_distance_is_clean_image(self) -> None:
        out = uw_render(self.rgb, torch.zeros(1, 1, 1), self.cfg)
        self.assertTrue(torch.equal(out, self.rgb))

    def test_open_water_is_backscatter_color(self) -> None:
        out = uw_render(self.rgb, torch.full((1, 1, 1), float("inf")), self.cfg)
        expected = [int(255 * b) for b in self.cfg.backscatter_value]  # truncation, as the uint8 cast
        self.assertEqual(out[0, 0, 0].tolist(), expected)

    def test_hand_computed_pixel(self) -> None:
        d = 10.0
        out = uw_render(self.rgb, torch.full((1, 1, 1), d), self.cfg)[0, 0, 0].tolist()
        for c in range(3):
            j = float(self.rgb[0, 0, 0, c])
            expected = j * math.exp(-d * self.cfg.atten_coeff[c]) + 255 * self.cfg.backscatter_value[c] * (
                1 - math.exp(-d * self.cfg.backscatter_coeff[c])
            )
            self.assertEqual(out[c], int(min(max(expected, 0), 255)))

    def test_converges_to_backscatter_with_distance(self) -> None:
        d = torch.linspace(0.0, 200.0, 401).view(1, -1, 1)
        rgb = self.rgb.expand(1, 401, 1, 3).clone().view(1, 401, 1, 3)
        out = uw_render(rgb, d, self.cfg)[0, :, 0].float()
        target = torch.tensor([255 * b for b in self.cfg.backscatter_value])
        err = (out - target).abs().sum(dim=-1)
        self.assertTrue(bool((err[1:] <= err[:-1] + 1.0).all()), "distance to backscatter color must not grow")
        self.assertLessEqual(float(err[-1]), 3.0)


class DVLGeometry(unittest.TestCase):
    def test_janus_pattern_angles(self) -> None:
        elev, rot = 22.5, 45.0
        _, dirs = dvl_janus_pattern(DVLJanusPatternCfg(elevation=elev, rotation=rot), "cpu")
        torch.testing.assert_close(dirs.norm(dim=-1), torch.ones(4))
        tilt = torch.rad2deg(torch.acos(-dirs[:, 2]))
        torch.testing.assert_close(tilt, torch.full((4,), elev), atol=1e-4, rtol=0)
        az = torch.rad2deg(torch.atan2(dirs[:, 1], dirs[:, 0])) % 360
        self.assertEqual(sorted(round(float(a)) % 360 for a in az), [45, 135, 225, 315])

    def test_flat_seabed_ranges(self) -> None:
        # flat seabed h below the sensor: each beam travels h / cos(elevation)
        elev, h = 22.5, 2.0
        _, dirs = dvl_janus_pattern(DVLJanusPatternCfg(elevation=elev), "cpu")
        origin = torch.tensor([[0.3, -1.0, 5.0]])
        t = h / (-dirs[:, 2])
        hits = origin[:, None, :] + dirs[None] * t[None, :, None]
        rng, hit = dvl_ranges_from_hits(hits, origin, 0.1, 10.0)
        torch.testing.assert_close(rng, torch.full((1, 4), h / math.cos(math.radians(elev))))
        self.assertTrue(bool(hit.all()))

    def test_misses_and_range_limits(self) -> None:
        origin = torch.zeros(1, 3)
        hits = torch.tensor([[[0.0, 0.0, -1.0], [float("inf")] * 3, [0.0, 0.0, -0.05], [0.0, 0.0, -50.0]]])
        rng, hit = dvl_ranges_from_hits(hits, origin, min_range=0.1, max_range=10.0)
        self.assertEqual(hit.tolist(), [[True, False, False, False]])
        self.assertAlmostEqual(float(rng[0, 0]), 1.0)
        self.assertTrue(bool(torch.isnan(rng[0, 1:]).all()))


class DVLRayOrigin(unittest.TestCase):
    def test_offset_is_rotated_into_world(self) -> None:
        # Isaac Lab 3.0 reports the sensor prim pose without cfg.offset; beams start at pos + R * offset
        q = quat_xyzw_from_axis_angle((0, 0, 1), math.pi / 2)
        origin = dvl_ray_origins_w(torch.tensor([[1.0, 2.0, 3.0]]), q, (0.1, 0.0, -0.2))
        torch.testing.assert_close(origin, torch.tensor([[1.0, 2.1, 2.8]]), atol=1e-6, rtol=0)


class DVLVelocity(unittest.TestCase):
    def test_world_velocity_in_body_frame(self) -> None:
        # vehicle yawed +90 deg moving along world +x -> body frame sees motion along -y
        q = quat_xyzw_from_axis_angle((0, 0, 1), math.pi / 2)
        v = dvl_body_velocity(torch.tensor([[1.0, 0.0, 0.0]]), q)
        torch.testing.assert_close(v, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=0)

    def test_lever_arm_pure_rotation(self) -> None:
        # yaw rate 1 rad/s, sensor 0.5 m ahead of the reference point: v_sensor = w x r = (0, 0.5, 0)
        q = quat_xyzw_from_axis_angle((0, 0, 1), 0.0)
        v = dvl_body_velocity(torch.zeros(1, 3), q, torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([0.5, 0.0, 0.0]))
        torch.testing.assert_close(v, torch.tensor([[0.0, 0.5, 0.0]]), atol=1e-6, rtol=0)

    def test_lever_arm_rotated_vehicle(self) -> None:
        # same, with the vehicle pitched: result is still omega_b x r_b in the body frame
        q = quat_xyzw_from_axis_angle((0, 1, 0), 0.4)
        w_b = torch.tensor([0.2, -0.3, 0.7])
        r_b = torch.tensor([0.1, 0.2, -0.3])
        import isaaclab.utils.math as mu

        w_w = mu.quat_apply(q, w_b[None])
        v = dvl_body_velocity(torch.zeros(1, 3), q, w_w, r_b)
        torch.testing.assert_close(v, torch.linalg.cross(w_b, r_b)[None], atol=1e-6, rtol=0)

    def test_beam_noise_covariance(self) -> None:
        # per-beam noise variance s2 -> body noise covariance s2 * T T^T
        torch.manual_seed(0)
        s2, n = 1e-4, 200_000
        T = janus_noise_transform(22.5)
        samples = (s2**0.5 * torch.randn(n, 4)) @ T.T
        emp = samples.T.cov()
        torch.testing.assert_close(emp, s2 * T @ T.T, atol=0.03 * s2 * float((T @ T.T).max()), rtol=0)


class BarometerModel(unittest.TestCase):
    def test_known_depths(self) -> None:
        z = torch.tensor([5.0, 0.0, -10.0])
        p = barometer_pressure_model(z, water_surface_z=0.0, water_density=1000.0, gravity=9.81)
        torch.testing.assert_close(p, torch.tensor([101325.0, 101325.0, 101325.0 + 98100.0]))

    def test_noise_variance(self) -> None:
        torch.manual_seed(1)
        p = barometer_pressure_model(torch.full((100_000,), -3.0), noise_cov=2400.0)
        self.assertAlmostEqual(float(p.var()), 2400.0, delta=60.0)
        self.assertAlmostEqual(float(p.mean()), 101325.0 + 1000.0 * 9.81 * 3.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
