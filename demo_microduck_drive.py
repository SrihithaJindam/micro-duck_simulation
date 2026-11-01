"""Drive MicroDuck yourself over an obstacle course, shove it, and measure how often it falls.

Reuses the walking demo's robot, motor model, and trained policies.

Run the interactive viewer:   python demo_microduck_drive.py
Run the headless push test:   python demo_microduck_drive.py --pushtest

Viewer keys (click the viewer window first):
  Up / Down      walk forward / backward (hold the speed until you press Stop)
  Left / Right   turn left / right
  S              stop (stand still)
  P              shove the duck sideways
  1 / 2 / 3      sit / stand / roll (policy switch)
"""

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from demo_microduck_walk import DECIMATION, Duck, _up, build_model_and_motors

WALK_SPEED = 0.3  # the walk policy stands still below about 0.3 m/s
TURN_RATE = 0.8
PUSH_SECONDS = 0.1

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262


def add_course(spec: mujoco.MjSpec) -> None:
    """Add low bumps and a gentle ramp in front of the duck (it starts facing +x)."""
    color = [0.85, 0.55, 0.2, 1.0]
    for x, half_height in [(0.3, 0.0015), (0.6, 0.003)]:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.03, 0.5, half_height],
            pos=[x, 0.0, half_height],
            rgba=color,
        )
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5, 0.5, 0.01],
        pos=[1.5, 0.0, 0.06],
        euler=[0.0, -6.0, 0.0],  # rises toward +x
        rgba=[0.3, 0.7, 0.4, 1.0],
    )


class Course:
    def __init__(self, with_course: bool = True):
        self.model, self.data, self.bam = build_model_and_motors(add_course if with_course else None)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("STAND").id)
        self.duck = Duck(self.model, self.data, self.bam)
        self.bam.reset(self.data.qpos)
        self.bam.q_target[:] = self.duck.default_pose
        mujoco.mj_forward(self.model, self.data)
        self.trunk = self.model.body("trunk_base").id
        self.vx = 0.0
        self.yaw = 0.0
        self.push_until = 0.0
        self.push_force = np.zeros(3)

    def set_walk(self, vx: float, yaw: float) -> None:
        self.vx, self.yaw = vx, yaw
        self.duck.set_behavior("walk", vx, yaw)

    def push(self, force_xy) -> None:
        """Shove the trunk with a horizontal force (newtons) for a short time."""
        self.push_force = np.array([force_xy[0], force_xy[1], 0.0])
        self.push_until = self.data.time + PUSH_SECONDS

    def step(self, i: int) -> None:
        if i % DECIMATION == 0:
            self.duck.policy_step()
        self.data.xfrc_applied[self.trunk, :3] = self.push_force if self.data.time < self.push_until else 0.0
        self.bam.update()
        mujoco.mj_step(self.model, self.data)

    def fallen(self) -> bool:
        return _up(self.model, self.data) < 0.5


def interactive() -> None:
    course = Course()
    course.set_walk(0.0, 0.0)
    rng = np.random.default_rng()

    def on_key(keycode: int) -> None:
        ch = chr(keycode).upper() if 0 < keycode < 128 else ""
        if keycode == KEY_UP:
            course.set_walk(WALK_SPEED, course.yaw)
        elif keycode == KEY_DOWN:
            course.set_walk(-WALK_SPEED, course.yaw)
        elif keycode == KEY_LEFT:
            course.set_walk(course.vx, TURN_RATE)
        elif keycode == KEY_RIGHT:
            course.set_walk(course.vx, -TURN_RATE)
        elif ch == "S":
            course.set_walk(0.0, 0.0)
        elif ch == "P":
            course.push(rng.choice([-1, 1]) * np.array([0.0, 1.0]) * 1.5)
        elif ch == "1":
            course.duck.set_behavior("sitstand", 1.0, 0.0)
        elif ch == "2":
            course.duck.set_behavior("sitstand", 0.0, 0.0)
        elif ch == "3":
            course.duck.set_behavior("roll", 0.0, 0.0)

    print(__doc__)
    with mujoco.viewer.launch_passive(course.model, course.data, key_callback=on_key) as viewer:
        i = 0
        while viewer.is_running():
            started = time.perf_counter()
            for _ in range(DECIMATION):  # one policy step, then one window update
                course.step(i)
                i += 1
            viewer.sync()
            time.sleep(max(0.0, DECIMATION * course.model.opt.timestep - (time.perf_counter() - started)))


def push_test(pushes_per_force: int = 8, forces=(20.0, 40.0, 80.0, 120.0, 160.0)) -> None:
    """Walk forward on flat ground, shove sideways, and count falls at each force."""
    rng = np.random.default_rng(0)
    print(f"Shoving the walking duck sideways for {PUSH_SECONDS}s, {pushes_per_force} times per force.")
    for force in forces:
        falls = 0
        for _ in range(pushes_per_force):
            course = Course(with_course=False)
            course.set_walk(WALK_SPEED, 0.0)
            for i in range(int(2.0 / course.model.opt.timestep)):  # settle into a walk
                course.step(i)
            course.push([0.0, rng.choice([-1, 1]) * force])
            for i in range(int(3.0 / course.model.opt.timestep)):
                course.step(i)
            falls += course.fallen()
        print(f"  {force:.1f} N: fell {falls}/{pushes_per_force}")


if __name__ == "__main__":
    if "--pushtest" in sys.argv:
        push_test()
    else:
        interactive()
