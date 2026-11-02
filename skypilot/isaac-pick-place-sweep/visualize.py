"""Replay measured Isaac Sim motion from one or more case folders in Rerun."""

import argparse
import json
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb


def log_case(folder):
    recording = json.loads((folder / "trajectory.json").read_text())
    scene = f"world/{folder.name}"
    names = recording["link_names"]
    arm = [names.index(f"panda_link{index}") for index in range(8)]
    hand = names.index("panda_hand")
    fingers = [names.index("panda_leftfinger"), names.index("panda_rightfinger")]
    frames = recording["frames"]
    target = recording["target_position_m"]
    size = recording["cube_size_m"]

    rr.log(scene, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        f"{scene}/target",
        rr.Boxes3D(
            centers=[target],
            sizes=[[size, size, size]],
            colors=[255, 210, 70],
            fill_mode=rr.components.FillMode.MajorWireframe,
            labels=["target"],
        ),
        static=True,
    )
    rr.log(
        f"{scene}/pickup",
        rr.Points3D(
            [recording["pick_position_m"]],
            radii=0.012,
            colors=[100, 220, 150],
            labels=["pickup"],
        ),
        static=True,
    )
    rr.log(
        f"{scene}/cube_path",
        rr.LineStrips3D(
            [[frame["cube_position_m"] for frame in frames]],
            radii=0.003,
            colors=[80, 200, 255],
        ),
        static=True,
    )

    for frame in frames:
        rr.set_time("simulation", duration=frame["time_s"])
        links = frame["link_positions_m"]
        strips = [[links[index] for index in [*arm, hand]]]
        strips.extend([[links[hand], links[index]] for index in fingers])
        rr.log(f"{scene}/arm", rr.LineStrips3D(strips, radii=0.025, colors=[215, 220, 230]))
        rr.log(
            f"{scene}/joints",
            rr.Points3D(
                [links[index] for index in arm],
                radii=0.032,
                colors=[255, 150, 75],
            ),
        )
        w, x, y, z = frame["cube_orientation_wxyz"]
        rr.log(
            f"{scene}/cube",
            rr.Boxes3D(
                centers=[frame["cube_position_m"]],
                sizes=[[size, size, size]],
                quaternions=[[x, y, z, w]],
                colors=[80, 160, 255],
            ),
        )
        rr.log(
            f"{scene}/phase",
            rr.Points3D(
                [[0, 0, 0.65]],
                radii=0,
                colors=[230, 230, 240],
                labels=[frame["phase"].replace("_", " ")],
            ),
        )

    return rrb.Spatial3DView(
        origin=scene,
        name=f"{folder.name} | pick X {recording['pick_position_m'][0]:.2f} → place Y {target[1]:.2f}",
        eye_controls=rrb.EyeControls3D(
            position=[0.7, 0.85, 0.6],
            look_target=[0, 0.15, 0.3],
            eye_up=[0, 0, 1],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="+", help="folders containing trajectory.json")
    parser.add_argument("--output", type=Path, default=Path("pick-place.rrd"))
    args = parser.parse_args()
    rr.init("isaac_pick_place", strict=True)
    rr.save(args.output)
    views = [log_case(folder) for folder in args.cases]
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(*views, grid_columns=2 if len(views) > 1 else 1),
            rrb.BlueprintPanel(expanded=False),
            rrb.SelectionPanel(expanded=False),
            rrb.TimePanel(timeline="simulation", play_state="paused"),
        )
    )
    rr.disconnect()
    print(f"Open with: rerun {args.output}")


if __name__ == "__main__":
    main()
