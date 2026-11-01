"""A Unitree Go2 robot dog dances in MuJoCo: bounce, sway, wiggle, play bow, nod, sit and wave.

The dance is choreographed by hand (no AI training): each move sets target angles
for the 12 leg joints, and a simple controller turns them into motor torques.
MuJoCo simulates the physics, so the dog really has to keep its balance.

Run:  python demo_dancing_dog.py
The dance repeats until you close the viewer window.
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = Path(__file__).parent / "assets" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"

KP = 60.0  # joint stiffness
KD = 2.0  # joint damping
BEAT_HZ = 1.6  # dance tempo (beats per second)
BLEND_S = 0.6  # seconds to blend smoothly from one move into the next

# Joint order: FL, FR, RL, RR legs; each leg is (hip, thigh, calf)
STAND = np.array([0.0, 0.9, -1.8] * 4)


def legs(fl, fr, rl, rr) -> np.ndarray:
    return np.array([*fl, *fr, *rl, *rr], dtype=float)


def beat(t: float, speed: float = 1.0) -> float:
    """-1..1 wave on the beat."""
    return np.sin(2 * np.pi * BEAT_HZ * speed * t)


def bounce(t):
    s = 0.5 * (1 - np.cos(2 * np.pi * BEAT_HZ * t))  # 0 -> 1 -> 0 each beat
    leg = (0.0, 0.9 + 0.35 * s, -1.8 - 0.6 * s)
    return legs(leg, leg, leg, leg)


def sway(t):
    side = 0.25 * beat(t, 0.5)
    leg = (side, 0.9, -1.8)
    return legs(leg, leg, leg, leg)


def wiggle(t):
    tail = 0.35 * beat(t, 1.5)
    front = (0.0, 0.9, -1.8)
    back = (tail, 0.9, -1.8)
    return legs(front, front, back, back)


def play_bow(t):
    dip = 0.5 * (1 - np.cos(2 * np.pi * BEAT_HZ * 0.5 * t))
    front = (0.0, 0.9 + 0.6 * dip, -1.8 - 0.8 * dip)
    back = (0.0, 0.9, -1.8)
    return legs(front, front, back, back)


def nod(t):
    # Only the front legs bob; moving the back legs too makes the dog walk backward
    s = 0.5 * (1 - np.cos(2 * np.pi * BEAT_HZ * t))
    front = (0.0, 0.9 + 0.3 * s, -1.8 - 0.5 * s)
    back = (0.0, 0.9, -1.8)
    return legs(front, front, back, back)


def sit_and_wave(t):
    front_left = (0.0, 0.6, -1.3)
    back = (0.0, 2.2, -2.7)
    if t < 1.5:  # sit down first
        front_right = (0.0, 0.6, -1.3)
    else:  # then wave the front right paw
        front_right = (-0.2, -0.4 + 0.35 * beat(t, 1.2), -1.0)
    return legs(front_left, front_right, back, back)


# (name, seconds, move)
DANCE = [
    ("Bounce", 5.0, bounce),
    ("Sway", 5.0, sway),
    ("Butt wiggle", 5.0, wiggle),
    ("Play bow", 5.0, play_bow),
    ("Nod", 4.0, nod),
    ("Sit and wave", 6.0, sit_and_wave),
    ("Stand up", 2.0, lambda t: STAND),
]


def run(show: bool = True, repeat: bool | None = None) -> list:
    """Run the dance. Returns how upright and steady the dog stayed in each move."""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    repeat = show if repeat is None else repeat
    results = []
    last_target = STAND.copy()
    viewer = mujoco.viewer.launch_passive(model, data) if show else None
    try:
        while True:
            for name, seconds, move in DANCE:
                print(f"Move: {name}")
                start_target = last_target.copy()
                min_up, min_height = 1.0, 1.0
                for i in range(int(seconds / model.opt.timestep)):
                    if viewer is not None and not viewer.is_running():
                        return results
                    started = time.perf_counter()

                    t = i * model.opt.timestep
                    blend = min(1.0, t / BLEND_S)
                    target = (1 - blend) * start_target + blend * move(t)
                    torque = KP * (target - data.qpos[7:]) - KD * data.qvel[6:]
                    data.ctrl[:] = np.clip(torque, low, high)
                    mujoco.mj_step(model, data)

                    min_up = min(min_up, float(data.xmat[1].reshape(3, 3)[2, 2]))
                    min_height = min(min_height, float(data.qpos[2]))
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - started)))
                last_target = target
                results.append(
                    (name, {"min_up": round(min_up, 2), "min_height_m": round(min_height, 3)})
                )
            if not repeat:
                return results
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    run(show=True)
