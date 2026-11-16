"""Compare OceanSim (Isaac Sim 5.1) and rov_lab port (Isaac Lab 3.0) renders of the shared scene.

Metrics (written to ``compare_metrics.json``) and a side-by-side figure (``compare_figure.png``):
  * camera intrinsics difference;
  * depth (``distance_to_camera``): geometry only, should agree regardless of renderer version;
  * clean RGB and underwater image: PSNR / SSIM / mean colour (renderer versions differ, so expect some gap);
  * model swap: rov_lab ``uw_render`` applied to OceanSim's own raw RGB + depth vs OceanSim's ``UW_Camera`` output
    (isolates the underwater model from renderer differences).

    PYTHONPATH=rov_lab/source <isaaclab-3.0-python> rov_lab/scripts/tools/oceansim_compare/compare_renders.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from rov_lab.assets.sensors import UWCameraCfg, uw_render

REPO = Path(__file__).resolve().parents[4]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on luminance with an 11x11 Gaussian window (sigma 1.5), standard constants."""

    def lum(x: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x.astype(np.float64))
        return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[None, None]

    x, y = lum(a), lum(b)
    g = torch.exp(-((torch.arange(11, dtype=torch.float64) - 5) ** 2) / (2 * 1.5**2))
    w = (g[:, None] * g[None, :]) / g.sum() ** 2
    w = w[None, None]
    mu_x, mu_y = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mu_x**2
    syy = F.conv2d(y * y, w) - mu_y**2
    sxy = F.conv2d(x * y, w) - mu_x * mu_y
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def image_stats(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "psnr_db": psnr(a, b),
        "ssim": ssim_gray(a, b),
        "mean_rgb_a": [float(v) for v in a.reshape(-1, 3).mean(0)],
        "mean_rgb_b": [float(v) for v in b.reshape(-1, 3).mean(0)],
        "mean_abs_diff": float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=REPO / "runs" / "oceansim_compare")
    ap.add_argument("--port_dir", type=str, default="port", help="sub-directory of --dir with the port render")
    ap.add_argument("--tag", type=str, default="", help="suffix for the metrics/figure file names")
    args = ap.parse_args()
    o = np.load(args.dir / "oceansim" / "oceansim_raw.npz")
    p = np.load(args.dir / args.port_dir / "port_raw.npz")
    o_uw = sorted((args.dir / "oceansim").glob("UW_image_*.png"))[-1]
    o_uw = np.asarray(Image.open(o_uw).convert("RGB"))
    p_uw = np.asarray(Image.open(args.dir / args.port_dir / "UW_image_port.png").convert("RGB"))
    o_rgb, p_rgb = o["rgba"][..., :3], p["rgb"][..., :3]
    o_d, p_d = o["depth"].astype(np.float64), p["depth"].astype(np.float64)

    m: dict = {}
    m["intrinsics_max_abs_diff"] = float(np.abs(o["K"] - p["K"]).max())
    valid = np.isfinite(o_d) & np.isfinite(p_d) & (o_d > 0) & (p_d > 0)
    dd = np.abs(o_d - p_d)[valid]
    rel = dd / o_d[valid]
    m["depth"] = {
        "valid_fraction_oceansim": float(np.isfinite(o_d).mean()),
        "valid_fraction_port": float(np.isfinite(p_d).mean()),
        "both_valid_fraction": float(valid.mean()),
        "abs_diff_median_m": float(np.median(dd)),
        "abs_diff_p99_m": float(np.quantile(dd, 0.99)),
        "rel_diff_median": float(np.median(rel)),
        "rel_diff_p99": float(np.quantile(rel, 0.99)),
        "frac_within_1cm": float((dd < 0.01).mean()),
        "frac_within_1pct": float((rel < 0.01).mean()),
    }
    m["raw_rgb"] = image_stats(o_rgb, p_rgb)
    m["underwater"] = image_stats(o_uw, p_uw)
    # global colour correction (per-channel 2nd-order polynomial, port -> OceanSim): what remains after removing
    # renderer-wide exposure / tone differences
    def fit_global(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        out = np.empty_like(src, dtype=np.float64)
        for c in range(3):
            x, y = src[..., c].astype(np.float64).ravel(), dst[..., c].astype(np.float64).ravel()
            coef = np.polyfit(x, y, 2)
            out[..., c] = np.polyval(coef, src[..., c].astype(np.float64))
        return np.clip(out, 0, 255).astype(np.uint8)

    m["raw_rgb_after_global_color_fit"] = image_stats(o_rgb, fit_global(p_rgb, o_rgb))
    m["underwater_after_global_color_fit"] = image_stats(o_uw, fit_global(p_uw, o_uw))
    swap = uw_render(torch.from_numpy(o_rgb.copy())[None], torch.from_numpy(o["depth"])[None], UWCameraCfg())[0].numpy()
    sd = np.abs(swap.astype(np.int32) - o_uw.astype(np.int32))
    m["model_swap_on_oceansim_inputs"] = {"max_abs_diff": int(sd.max()), "frac_exact": float((sd == 0).mean()),
                                          "frac_within_1": float((sd <= 1).mean()), "psnr_db": psnr(swap, o_uw)}
    (args.dir / f"compare_metrics{args.tag}.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))

    # figure: row 1 clean RGB (OceanSim | port | |diff| x4), row 2 underwater (OceanSim | port | depth |diff| heatmap)
    h, w = o_rgb.shape[:2]
    def diff_img(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.clip(np.abs(a.astype(np.int32) - b.astype(np.int32)) * 4, 0, 255).astype(np.uint8)
    dmap = np.zeros((h, w), np.float64)
    dmap[valid] = np.abs(o_d - p_d)[valid]
    heat = np.clip(dmap / 0.05, 0, 1)  # 0..5 cm
    heat_rgb = (np.stack([heat, np.zeros_like(heat), 1 - heat], -1) * 255).astype(np.uint8)
    heat_rgb[~valid] = 40
    tiles = [[o_rgb, p_rgb, diff_img(o_rgb, p_rgb)], [o_uw, p_uw, heat_rgb]]
    labels = [["OceanSim (Isaac Sim 5.1) clean RGB", "rov_lab port (Isaac Lab 3.0) clean RGB", "|diff| x4"],
              ["OceanSim UW_Camera underwater", "rov_lab port uw_render underwater", "depth |diff| (blue 0 -> red >=5 cm)"]]
    pad = 28
    canvas = Image.new("RGB", (3 * w + 20, 2 * (h + pad) + 10), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for r in range(2):
        for c in range(3):
            x0, y0 = c * (w + 10), r * (h + pad + 5)
            draw.text((x0 + 4, y0 + 4), labels[r][c], fill="black", font=font)
            canvas.paste(Image.fromarray(tiles[r][c]), (x0, y0 + pad))
    canvas.save(args.dir / f"compare_figure{args.tag}.png")
    print(f"figure -> {args.dir / f'compare_figure{args.tag}.png'}")


if __name__ == "__main__":
    main()
