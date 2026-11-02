"""Run one headless Franka pick-and-place case and publish its result."""

import argparse
import json
import math
import os
import re
from pathlib import Path

PLACEMENT_TOLERANCE_M = 0.08
CUBE_HEIGHT_M = 0.0515
PHYSICS_DT = 1 / 60


def evaluate(controller_done, final_position, target_position):
    positions = list(final_position) + list(target_position)
    if not all(math.isfinite(value) for value in positions):
        return {"xy_error_m": None, "z_error_m": None, "success": False}
    xy_error = math.dist(final_position[:2], target_position[:2])
    z_error = abs(final_position[2] - target_position[2])
    return {
        "xy_error_m": xy_error,
        "z_error_m": z_error,
        "success": (
            controller_done
            and xy_error <= PLACEMENT_TOLERANCE_M
            and z_error <= PLACEMENT_TOLERANCE_M
        ),
    }


def record_frame(task, robot_links, step):
    cube_positions, cube_orientations = task.cubes[0].get_world_poses()
    return {
        "time_s": step * PHYSICS_DT,
        "phase": task.status()["phase"],
        "cube_position_m": cube_positions.numpy()[0].tolist(),
        "cube_orientation_wxyz": cube_orientations.numpy()[0].tolist(),
        "link_positions_m": robot_links.get_world_poses()[0].numpy().tolist(),
    }


def simulate(app, pick_x, place_y, max_steps, recording_path=None):
    # Isaac extensions must be imported after the app starts.
    import isaacsim.core.experimental.utils.app as app_utils

    app_utils.enable_extension("isaacsim.robot_motion.examples")

    import numpy as np
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.robot_motion.examples.manipulation import PickPlaceTask

    cube_height = CUBE_HEIGHT_M / 2
    pick_position = [pick_x, 0.2, cube_height]
    target_position = [-0.4, place_y, cube_height]

    # Set both goals before creating the scene; Isaac captures them at reset.
    task = PickPlaceTask(cube_positions=[tuple(pick_position)])
    task.place_position = np.asarray(target_position, dtype=np.float32)
    task.setup_scene()

    SimulationManager.setup_simulation(dt=PHYSICS_DT, device="cpu")
    SimulationManager.get_physics_scenes()[0].set_enabled_gpu_dynamics(False)
    app_utils.play()
    app_utils.update_app(steps=20)
    task.initialize()
    task.reset()

    frames = []
    if recording_path is not None:
        from isaacsim.core.experimental.prims import RigidPrim

        robot_links = RigidPrim(task.scenario.articulation.link_paths[0])
        frames.append(record_frame(task, robot_links, 0))

    for step in range(1, max_steps + 1):
        app.update()
        task.step(PHYSICS_DT)
        if recording_path is not None and step % 6 == 0:
            frames.append(record_frame(task, robot_links, step))
        if task.is_done or task.failed:
            break
    # Let the cube settle before measuring its final pose.
    app_utils.update_app(steps=30)

    if recording_path is not None:
        frames.append(record_frame(task, robot_links, step + 30))
        recording = {
            "link_names": task.scenario.articulation.link_names,
            "pick_position_m": pick_position,
            "target_position_m": target_position,
            "cube_size_m": CUBE_HEIGHT_M,
            "frames": frames,
        }
        recording_path.write_text(json.dumps(recording, allow_nan=False) + "\n")

    cube_positions = task.cubes[0].get_world_poses()[0].numpy()
    final_position = cube_positions[0].tolist()
    controller_done = bool(task.is_done and not task.failed)
    task.cleanup()
    app_utils.stop()
    result = {
        "isaac_sim_version": "6.1.0",
        "robot": "Franka Panda",
        "pick_position_m": pick_position,
        "target_position_m": target_position,
        "final_cube_position_m": [
            value if math.isfinite(value) else None for value in final_position
        ],
        "controller_done": controller_done,
        "steps": step,
        "max_steps": max_steps,
        **evaluate(controller_done, final_position, target_position),
    }
    return result


def check_s3_environment():
    required = (
        "S3_BUCKET",
        "S3_ENDPOINT_URL",
        "AWS_DEFAULT_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    )
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError(f"missing S3 environment variables: {', '.join(missing)}")


def upload(result_path, run_id, case_id, recording_path=None):
    import boto3

    check_s3_environment()
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "isaac-pick-place").strip("/")
    case_prefix = f"{prefix}/{run_id}/{case_id}"
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )
    if recording_path is not None:
        client.upload_file(str(recording_path), bucket, f"{case_prefix}/trajectory.json")
    client.upload_file(str(result_path), bucket, f"{case_prefix}/result.json")
    # Readers use COMPLETE to tell that result.json finished uploading.
    client.put_object(Bucket=bucket, Key=f"{case_prefix}/COMPLETE", Body=b"")
    print(f"published s3://{bucket}/{case_prefix}/result.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--pick-x", type=float, required=True)
    parser.add_argument("--place-y", type=float, required=True)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "isaac-results")
    parser.add_argument(
        "--local", action="store_true", help="keep result locally without S3 upload"
    )
    parser.add_argument(
        "--record-motion",
        action="store_true",
        help="save measured robot and cube poses for Rerun replay",
    )
    args = parser.parse_args()
    for name in ("run_id", "case_id"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must use letters, digits, _ or -")
    if not math.isfinite(args.pick_x) or not math.isfinite(args.place_y) or args.max_steps < 1:
        parser.error("positions must be finite and --max-steps must be positive")
    if not args.local:
        try:
            check_s3_environment()
        except ValueError as exc:
            parser.error(str(exc))

    from isaacsim import SimulationApp

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recording_path = args.output_dir / "trajectory.json" if args.record_motion else None
    app = SimulationApp({"headless": True})
    result = simulate(app, args.pick_x, args.place_y, args.max_steps, recording_path)
    result.update({"run_id": args.run_id, "case_id": args.case_id})
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    if not args.local:
        upload(result_path, args.run_id, args.case_id, recording_path)
    # Isaac's fast shutdown can exit Python, so publish before closing the app.
    app.close()


if __name__ == "__main__":
    main()
