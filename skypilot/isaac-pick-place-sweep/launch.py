"""Submit one Isaac Sim managed job per parameter combination to SkyPilot."""

import argparse
import json
import math
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml


def expand_sweep(config):
    if not isinstance(config, dict) or set(config) != {"pick_x", "place_y"}:
        raise ValueError("sweep.json must contain only pick_x and place_y")
    for axis in ("pick_x", "place_y"):
        values = config[axis]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{axis} must be a nonempty list of finite numbers")
        for value in values:
            is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not is_number or not math.isfinite(value):
                raise ValueError(f"{axis} must be a nonempty list of finite numbers")
    cases = []
    for pick_x in config["pick_x"]:
        for place_y in config["place_y"]:
            cases.append({"pick_x": pick_x, "place_y": place_y})
    return cases


def job_config(options, run_id, case_id, point):
    config = yaml.safe_load(Path(__file__).with_name("task.yaml").read_text())
    config["name"] = f"isaac-{run_id.lower()}-{case_id}"
    config["resources"]["infra"] = f"nebius/{options.region}"
    config["envs"].update(
        {
            "IMAGE": options.image,
            "S3_BUCKET": options.bucket,
            "S3_ENDPOINT_URL": f"https://storage.{options.region}.nebius.cloud",
            "AWS_DEFAULT_REGION": options.region,
            "S3_PREFIX": options.prefix.strip("/"),
            "RUN_ID": run_id,
            "CASE_ID": case_id,
            "PICK_X": str(point["pick_x"]),
            "PLACE_Y": str(point["place_y"]),
            "RECORD_MOTION": "1" if options.record_motion else "0",
        }
    )
    return config


def read_secrets():
    names = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    registry_names = ["REGISTRY_SERVER", "REGISTRY_USERNAME", "REGISTRY_PASSWORD"]
    if any(os.environ.get(name) for name in registry_names):
        names.extend(registry_names)
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise ValueError(f"set these environment variables: {', '.join(missing)}")
    return {name: os.environ[name] for name in names}


def submit_jobs(configs, manifest, secrets):
    import sky

    # Validate every task before accepting any jobs.
    tasks = [
        sky.Task.from_yaml_config(config, secrets_overrides=list(secrets.items()))
        for config in configs
    ]
    manifest.parent.mkdir(parents=True)
    jobs = []
    for task in tasks:
        env = task.envs
        job = {
            "case_id": env["CASE_ID"],
            "params": {"pick_x": float(env["PICK_X"]), "place_y": float(env["PLACE_Y"])},
            "job_name": task.name,
            "s3_uri": f"s3://{env['S3_BUCKET']}/{env['S3_PREFIX']}/{env['RUN_ID']}/{env['CASE_ID']}/",
            "request_id": sky.jobs.launch(task, name=task.name),
        }
        jobs.append(job)
        # Save the accepted request before waiting, so an interruption is traceable.
        manifest.write_text(json.dumps(jobs, indent=2) + "\n")
        job_ids, _ = sky.get(job["request_id"])
        if not job_ids or len(job_ids) != 1:
            raise RuntimeError(
                "expected one managed job ID; inspect the saved request before retrying"
            )
        job["job_id"] = job_ids[0]
        manifest.write_text(json.dumps(jobs, indent=2) + "\n")
        print(f"{job['case_id']} job_id={job['job_id']} {job['s3_uri']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="image built from this recipe's Dockerfile")
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--region", default="eu-north1", help="region for both Compute and Object Storage"
    )
    parser.add_argument("--prefix", default="isaac-pick-place")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("sweep.json"))
    parser.add_argument(
        "--single", action="store_true", help="submit only the first point in sweep.json"
    )
    parser.add_argument("--record-motion", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print tasks without reading secrets or contacting SkyPilot",
    )
    options = parser.parse_args()
    if not options.prefix.strip("/"):
        parser.error("--prefix must contain a path component")
    try:
        points = expand_sweep(json.loads(options.config.read_text()))
        secrets = {} if options.dry_run else read_secrets()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if options.single:
        points = points[:1]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    configs = [
        job_config(options, run_id, f"case-{index:03d}", point)
        for index, point in enumerate(points)
    ]
    print(f"run_id={run_id} jobs={len(configs)}", flush=True)
    if options.dry_run:
        print(yaml.safe_dump_all(configs, sort_keys=False))
        return
    manifest = Path(__file__).with_name("runs") / run_id / "jobs.json"
    print(f"job manifest: {manifest}", flush=True)
    try:
        submit_jobs(configs, manifest, secrets)
    except Exception as exc:
        message = str(exc)
        for value in secrets.values():
            message = message.replace(value, "[redacted]")
        raise SystemExit(
            f"submission stopped: {message}; inspect {manifest} and sky jobs queue before retrying"
        ) from None


if __name__ == "__main__":
    main()
