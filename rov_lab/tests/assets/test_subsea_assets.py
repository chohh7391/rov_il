"""Checks for the subsea work-site layout, the valve-panel USD and the third-party license records.

The USD checks run only when the assets have been built (``build_subsea_assets.py``); the layout and license checks
always run. No simulator needed:
    PYTHONPATH=rov_lab/source <isaaclab-3.0-python> -m unittest discover -s rov_lab/tests -t rov_lab -v
"""

from __future__ import annotations

import importlib.util
import json
import math
import unittest
from pathlib import Path

REPO: Path = Path(__file__).resolve().parents[3]
LAYOUT_PATH: Path = REPO / "rov_lab" / "source" / "rov_lab" / "assets" / "scenes" / "work_site_layout.py"
THIRD_PARTY: Path = REPO / "rov_lab" / "assets_src" / "third_party"

# rov_bop_panel valves from uuv_gazebo_worlds/models/rov_bop_panel/model.sdf: joint position (panel frame), yaw (rad)
SDF_VALVES: dict[str, tuple[tuple[float, float, float], float]] = {
    "valve_1": ((-0.15, -0.2, 0.18), 1.57),
    "valve_2": ((0.15, -0.2, 0.18), 1.57),
    "valve_3": ((-0.15, 0.0, 0.18), 1.57),
    "valve_4": ((0.0, -0.2, 0.14), 0.0),
    "valve_5": ((0.0, 0.0, 0.14), 0.0),
}


def _load_layout():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("work_site_layout", LAYOUT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _load_layout()


class WorkSiteLayout(unittest.TestCase):
    def test_panel_rotation_is_proper(self) -> None:
        r = L.panel_rotation_matrix()
        for i in range(3):
            for j in range(3):
                dot = sum(r[k][i] * r[k][j] for k in range(3))
                self.assertAlmostEqual(dot, 1.0 if i == j else 0.0, places=9)
        det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
               + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
        self.assertAlmostEqual(det, 1.0, places=9)

    def test_valve_axes_face_the_rov(self) -> None:
        r = L.panel_rotation_matrix()
        self.assertEqual(tuple(round(r[k][2], 9) for k in range(3)), (-1.0, 0.0, 0.0))  # panel z -> -X
        self.assertEqual(tuple(round(r[k][1], 9) for k in range(3)), (0.0, 0.0, 1.0))  # panel y -> +Z (rows level)

    def test_valves_in_front_of_frame_and_reachable_height(self) -> None:
        for name, (pos, _) in SDF_VALVES.items():
            p = L.panel_point_in_site(pos)
            self.assertLess(p[0], L.FRAME_FRONT_X, name)
            self.assertGreater(p[0], L.ROV_STATION_POS[0], name)
            self.assertTrue(0.4 < p[2] < 1.2, f"{name} at z={p[2]:.2f}")

    def test_rocks_clear_of_approach_corridor(self) -> None:
        for rid, x, y, _, size in L.ROCKS:
            in_corridor = (-5.0 < x < 1.5) and (-1.0 < y < 2.5)
            self.assertFalse(in_corridor, f"rock {rid} at ({x}, {y})")
            self.assertTrue(1 <= rid <= 7 and size > 0)


class ThirdPartyRecords(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads((THIRD_PARTY / "manifest.json").read_text())
        self.attribution = (THIRD_PARTY / "ATTRIBUTION.md").read_text()
        self.notice = (THIRD_PARTY / "NOTICE").read_text()

    def test_only_permissive_licenses(self) -> None:
        allowed = {"Apache-2.0", "CC-BY-4.0", "CC0-1.0"}
        for group in ("fuel", "github", "polyhaven"):
            for entry in self.manifest[group]:
                self.assertIn(entry["license"], allowed, entry["id"])

    def test_every_cc_by_asset_is_attributed(self) -> None:
        for entry in self.manifest["fuel"]:
            if entry["license"] == "CC-BY-4.0":
                stem = entry["name"].rsplit(" ", 1)[0] if entry["name"].startswith("Ocean Rock") else entry["name"]
                self.assertIn(stem, self.attribution, entry["id"])

    def test_apache_sources_have_notices(self) -> None:
        self.assertTrue((THIRD_PARTY / "LICENSE-Apache-2.0.txt").exists())
        for entry in self.manifest["github"]:
            self.assertEqual(entry["license"], "Apache-2.0")
            self.assertIn(entry["repo"], self.notice)
            self.assertIn(entry["commit"], self.notice)

    def test_texturelib_texture_not_fetched(self) -> None:
        files = [f for e in self.manifest["github"] for f in e["files"]]
        self.assertFalse(any(f.endswith("metal_bare_0045_01.jpg") for f in files))


class ValvePanelUsd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import os

        root = Path(os.environ.get("ROV_LAB_ASSETS_ROOT", REPO / "rov_lab" / "assets"))
        cls.usd = root / "objects" / "rov_bop_panel" / "rov_bop_panel.usda"
        if not cls.usd.exists():
            raise unittest.SkipTest(f"panel asset not built: {cls.usd}")
        from pxr import Usd, UsdPhysics

        cls.Usd, cls.UsdPhysics = Usd, UsdPhysics
        cls.stage = Usd.Stage.Open(str(cls.usd))

    def test_articulation_root_and_masses(self) -> None:
        panel = self.stage.GetPrimAtPath("/rov_bop_panel/panel")
        self.assertTrue(panel.HasAPI(self.UsdPhysics.ArticulationRootAPI))
        self.assertAlmostEqual(self.UsdPhysics.MassAPI(panel).GetMassAttr().Get(), 10.0)
        for name in SDF_VALVES:
            valve = self.stage.GetPrimAtPath(f"/rov_bop_panel/{name}")
            self.assertTrue(valve.HasAPI(self.UsdPhysics.RigidBodyAPI), name)
            self.assertAlmostEqual(self.UsdPhysics.MassAPI(valve).GetMassAttr().Get(), 0.5, msg=name)

    def test_valve_joints_match_sdf(self) -> None:
        joints = {p.GetName(): self.UsdPhysics.RevoluteJoint(p) for p in self.stage.Traverse()
                  if p.IsA(self.UsdPhysics.RevoluteJoint)}
        self.assertEqual(sorted(joints), [f"{v}_joint" for v in sorted(SDF_VALVES)])
        for name, (pos, yaw) in SDF_VALVES.items():
            j = joints[f"{name}_joint"]
            self.assertEqual(j.GetAxisAttr().Get(), "Z")
            self.assertEqual((j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()), (-90.0, 0.0))
            p0 = j.GetLocalPos0Attr().Get()
            for a, b in zip(p0, pos):
                self.assertAlmostEqual(a, b, places=5)
            q = j.GetLocalRot0Attr().Get()
            self.assertAlmostEqual(2.0 * math.atan2(q.GetImaginary()[2], q.GetReal()), yaw, places=4)
            self.assertEqual([str(t) for t in j.GetBody0Rel().GetTargets()], ["/rov_bop_panel/panel"])

    def test_paddles_have_box_colliders(self) -> None:
        for name in SDF_VALVES:
            paddle = self.stage.GetPrimAtPath(f"/rov_bop_panel/{name}/collisions/paddle")
            self.assertTrue(paddle.IsValid() and paddle.HasAPI(self.UsdPhysics.CollisionAPI), name)


if __name__ == "__main__":
    unittest.main()
