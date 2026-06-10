"""CLI for converting pick-stone HDF5 demos to LeRobot Dataset v3."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_TARGET_COLOR_KEY, DEFAULT_TASK_DESCRIPTION, PickStoneDatasetConfig
from .converter import convert_pick_stone_hdf5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id, e.g. rov_il/pick_stone.")
    parser.add_argument("--output-root", required=True, type=Path, help="Local output directory for v3 data.")
    parser.add_argument("--hdf5-files", required=True, nargs="+", type=Path, help="Input Isaac Lab HDF5 files.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="fixed_rov_single_arm")
    parser.add_argument(
        "--task",
        default=None,
        help="Fallback task string used only with --fixed-task-fallback.",
    )
    parser.add_argument(
        "--target-color-key",
        default=DEFAULT_TARGET_COLOR_KEY,
        help="HDF5 key containing the per-episode target color id.",
    )
    parser.add_argument(
        "--fixed-task-fallback",
        action="store_true",
        help="Use --task when target-color metadata is missing.",
    )
    parser.add_argument("--action-key", default="actions")
    parser.add_argument("--state-key", default="obs/joint_pos")
    parser.add_argument("--camera-keys", nargs="+", default=["front", "wrist", "sonar"])
    parser.add_argument("--skip-first-frames", type=int, default=5)
    parser.add_argument("--min-episode-frames", type=int, default=10)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true", help="Store image files instead of encoded videos.")
    parser.add_argument("--vcodec", default="auto")
    parser.add_argument("--streaming-encoding", action="store_true")
    args = parser.parse_args()

    cfg = PickStoneDatasetConfig(
        repo_id=args.repo_id,
        output_root=args.output_root,
        hdf5_files=tuple(args.hdf5_files),
        fps=args.fps,
        robot_type=args.robot_type,
        task=args.task or DEFAULT_TASK_DESCRIPTION,
        action_key=args.action_key,
        state_key=args.state_key,
        camera_keys=tuple(args.camera_keys),
        skip_first_frames=args.skip_first_frames,
        min_episode_frames=args.min_episode_frames,
        require_success=args.require_success,
        overwrite=args.overwrite,
        use_videos=not args.no_videos,
        vcodec=args.vcodec,
        streaming_encoding=args.streaming_encoding,
        target_color_key=args.target_color_key,
        allow_fixed_task_fallback=args.fixed_task_fallback,
    )
    summary = convert_pick_stone_hdf5(cfg)
    print(
        "Converted pick-stone dataset: "
        f"{summary.episodes_written} episodes, {summary.frames_written} frames "
        f"({summary.episodes_skipped} skipped) -> {summary.output_root}"
    )


if __name__ == "__main__":
    main()
