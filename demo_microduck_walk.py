"""MicroDuck stands, looks around, walks, circles, turns, sits, stands, and rolls in MuJoCo.

The behaviors come from Pollen Robotics' trained policies on Hugging Face
(pollen-robotics/microduck-policies):
- velstand.onnx: walking, turning, standing still, and getting up after a fall
- alpha_sitstand.onnx: sitting down and standing back up
- roulade.onnx: a forward roll over the head

Each policy reads the same 61 numbers from the simulated robot (body motion,
joint positions, last action, and a command) and returns 14 joint targets,
50 times per second. The motors use the BAM M6 actuator model, the same motor
model the policies were trained with.

Run:  python demo_microduck_walk.py
The routine repeats until you close the viewer window.
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
from bam.model import load_model
from bam.mujoco import MujocoController

ROOT = Path(__file__).parent
SCENE = ROOT / "assets" / "microduck_rl" / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene.xml"
POLICY_DIR = ROOT / "assets" / "policies"
POLICY_FILES = {
    "walk": "velstand.onnx",
    "sitstand": "alpha_sitstand.onnx",
    "roll": "roulade.onnx",
}

# Settings copied from scripts/infer_policy.py in the microduck_rl repo
TIMESTEP = 0.005
DECIMATION = 4  # 4 physics steps of 5 ms = 50 Hz policy
KP_FIRMWARE = 200.0
BATTERY_VOLTS = 7.4
VOLTAGE_SAG_GAIN = 0.1
VOLTAGE_MIN = 6.0
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)

# Command layout (13 numbers): [vx, vy, yaw_rate, neck_pitch, head_pitch, head_yaw, head_roll, body(6)]
HEAD_YAW_SLOT = 5
SIT_FLAG_SLOT = 0  # the sitstand policy reads vx as a posture flag: 1 = sit, 0 = stand.
# Switch the flag instantly: the policy smooths the motion itself, and ramping the flag tips it over.

# (label, policy, seconds, vx m/s, yaw rad/s, head-yaw swing rad)
# The walk policy stands still below about 0.3 m/s, so walking uses 0.3.
ROUTINE = [
    ("Stand still", "walk", 3.0, 0.0, 0.0, 0.0),
    ("Look left and right", "walk", 6.0, 0.0, 0.0, 0.8),
    ("Walk forward", "walk", 5.0, 0.3, 0.0, 0.0),
    ("Walk in a circle", "walk", 12.0, 0.3, 0.5, 0.0),
    ("Turn in place", "walk", 6.0, 0.0, 1.0, 0.0),
    ("Stop and settle", "walk", 4.0, 0.0, 0.0, 0.0),
    ("Sit down", "sitstand", 5.0, 1.0, 0.0, 0.0),
    ("Stand up", "sitstand", 5.0, 0.0, 0.0, 0.0),
    ("Forward roll", "roll", 2.0, 0.0, 0.0, 0.0),
    ("Recover and stand", "walk", 5.0, 0.0, 0.0, 0.0),
]


def quat_rotate_inverse(quat, vec):
    """Rotate a vector by the inverse of a quaternion [w, x, y, z]."""
    w = quat[0]
    xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def build_model_and_motors(edit_spec=None):
    """Load the robot with the BAM motor model on each joint. Mirrors infer_policy.py.

    edit_spec, if given, is called with the MjSpec before compiling (to add obstacles etc.).
    """
    bam_model = load_model(motor_name="xl330", model="m6")
    bam_model.actuator.kp = KP_FIRMWARE
    bam_model.actuator.vin = BATTERY_VOLTS
    bam_model.actuator.max_current = None

    kt = bam_model.kt.value
    R = bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R

    spec = mujoco.MjSpec.from_file(str(SCENE))
    motor_names = []
    for act in spec.actuators:
        target = act.target
        target_name = target.name if hasattr(target, "name") else str(target)
        if target_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        motor_names.append(act.name)
        for joint in spec.joints:
            if joint.name == target_name:
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                joint.solref_friction = STIFF_SOLREF_FRICTION
                joint.solimp_friction = STIFF_SOLIMP_FRICTION
                break

    if edit_spec is not None:
        edit_spec(spec)
    model = spec.compile()
    model.opt.timestep = TIMESTEP
    data = mujoco.MjData(model)
    bam = MujocoController(
        bam_model, motor_names, model, data, vin_drop_gain=VOLTAGE_SAG_GAIN, vin_min=VOLTAGE_MIN
    )
    return model, data, bam


class Duck:
    """Builds the policies' 61-number input and sends their 14 joint targets to the motors."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, bam: MujocoController):
        self.model = model
        self.data = data
        self.bam = bam
        self.sessions = {
            name: ort.InferenceSession(str(POLICY_DIR / file)) for name, file in POLICY_FILES.items()
        }
        self.active = "walk"

        meta = self.sessions["walk"].get_modelmeta().custom_metadata_map
        self.default_pose = np.array(
            [float(x) for x in meta["default_joint_pos"].split(",")][:14], dtype=np.float32
        )
        self.action_scale = float(meta.get("action_scale", "1.0"))

        self.joint_qpos = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(14)]
        self.joint_qvel = [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(14)]
        gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.gyro_adr = int(model.sensor_adr[gyro_id])
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

        self.last_action = np.zeros(14, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)

    def set_behavior(self, policy: str, vx: float, yaw_rate: float) -> None:
        self.active = policy
        self.command[:] = 0.0
        if policy == "walk":
            self.command[0] = vx
            self.command[2] = yaw_rate
        elif policy == "sitstand":
            self.command[SIT_FLAG_SLOT] = vx
        # The roll policy was trained with an all-zero command

    def set_head_yaw(self, angle: float) -> None:
        if self.active == "walk":
            self.command[HEAD_YAW_SLOT] = angle

    def observe(self) -> np.ndarray:
        gyro = self.data.sensordata[self.gyro_adr:self.gyro_adr + 3]
        gravity = quat_rotate_inverse(
            self.data.xquat[self.trunk].astype(np.float32), np.array([0.0, 0.0, -1.0])
        )
        joint_pos = self.data.qpos[self.joint_qpos] - self.default_pose
        joint_vel = self.data.qvel[self.joint_qvel]
        obs = np.concatenate([gyro, gravity, joint_pos, joint_vel, self.last_action, self.command])
        return obs.astype(np.float32).reshape(1, -1)

    def policy_step(self) -> None:
        """Ask the active policy for an action and set the motor targets."""
        session = self.sessions[self.active]
        inputs = {session.get_inputs()[0].name: self.observe()}
        action = session.run([session.get_outputs()[0].name], inputs)[0]
        action = action.squeeze(0).astype(np.float32)
        self.last_action = action
        self.bam.q_target[:] = self.default_pose + action * self.action_scale


def run(show: bool = True, routine: list | None = None, repeat: bool | None = None) -> list:
    """Run the routine. Returns the duck's state after each behavior, for checking without a window."""
    model, data, bam = build_model_and_motors()
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    duck = Duck(model, data, bam)
    bam.reset(data.qpos)
    bam.q_target[:] = duck.default_pose
    mujoco.mj_forward(model, data)

    routine = routine or ROUTINE
    repeat = show if repeat is None else repeat
    results = []
    viewer = mujoco.viewer.launch_passive(model, data) if show else None
    try:
        while True:
            for label, policy, seconds, vx, yaw, head_swing in routine:
                duck.set_behavior(policy, vx, yaw)
                print(f"{label} ({policy} policy)")
                start_time = data.time
                stats = {"turned_deg": 0.0, "min_up": 1.0, "head_yaw_max": 0.0}
                last_heading = _heading(model, data)
                for i in range(int(seconds / model.opt.timestep)):
                    if viewer is not None and not viewer.is_running():
                        return results
                    started = time.perf_counter()
                    if i % DECIMATION == 0:
                        t = data.time - start_time
                        if head_swing:
                            duck.set_head_yaw(head_swing * np.sin(2 * np.pi * t / 3.0))
                        duck.policy_step()
                        heading = _heading(model, data)
                        stats["turned_deg"] += (heading - last_heading + 180.0) % 360.0 - 180.0
                        last_heading = heading
                        stats["min_up"] = min(stats["min_up"], _up(model, data))
                        stats["head_yaw_max"] = max(
                            stats["head_yaw_max"], abs(float(data.qpos[duck.joint_qpos[7]]))
                        )
                    bam.update()  # BAM computes the motor torque for this physics step
                    mujoco.mj_step(model, data)
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - started)))
                state = _state(model, data)
                state.update({k: round(v, 2) for k, v in stats.items()})
                results.append((label, state))
            if not repeat:
                return results
    finally:
        if viewer is not None:
            viewer.close()


def _heading(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    rot = data.xmat[model.body("trunk_base").id].reshape(3, 3)
    return float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))


def _up(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """1.0 = upright, 0 = on its side, -1.0 = upside down."""
    return float(data.xmat[model.body("trunk_base").id].reshape(3, 3)[2, 2])


def _state(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    rot = data.xmat[trunk].reshape(3, 3)
    return {
        "x_m": round(float(data.qpos[0]), 3),
        "y_m": round(float(data.qpos[1]), 3),
        "heading_deg": round(float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))), 1),
        "height_m": round(float(data.qpos[2]), 3),
        "upright": bool(rot[2, 2] > 0.9),
    }


if __name__ == "__main__":
    run(show=True)
