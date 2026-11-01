"""A small wheeled robot finds its own way through a random 3D maze in MuJoCo.

The robot has five distance sensors (MuJoCo rangefinders, like a cheap lidar). At the center of
each cell it applies the right-hand rule: turn right if the right side is open, else go straight,
else turn left, else turn around. It has no map, only its sensors plus its own position and
heading (like wheel odometry and a compass). A random maze without loops can always be solved
this way. A trail of yellow dots shows where it has been.

Run the viewer:     python demo_maze_robot.py
Headless self-test: python demo_maze_robot.py --selftest

Viewer keys (click the viewer window first):
  N   build a new random maze
  R   put the robot back at the start of this maze
After the robot reaches the green goal, a new maze is built automatically.
"""

import math
import random
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

CELLS = 6
CELL = 0.5  # meters
WALL_THICK = 0.03
WALL_HEIGHT = 0.2
HIDDEN_Z = -5.0  # unused wall pieces and trail dots wait under the floor

WHEEL_R = 0.03
HALF_TRACK = 0.075
TRAIL_DOTS = 600
TRAIL_EVERY = 0.1  # seconds of simulated time between dots

SENSOR_ANGLES = {"front": 0, "front_left": 45, "front_right": -45, "left": 90, "right": -90}
CUTOFF = 2.0

CONTROL_DT = 0.01

SPEED = 0.28
WALL_GAP = 0.16  # sensor reading when the robot is in the middle of a corridor
LATERAL_GAIN = 2.5  # rad of heading offset per meter off the corridor middle
HEADING_GAIN = 4.0
TURN_RATE = 1.8
OPEN_GAP = 0.32  # a sensor reading beyond this means there is no wall on that side


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def wall_pieces():
    """Every possible wall segment on the grid: ('h', row, col) runs along x, ('v', row, col) along y."""
    for i in range(CELLS + 1):
        for j in range(CELLS):
            yield "h", i, j
    for i in range(CELLS):
        for j in range(CELLS + 1):
            yield "v", i, j


def piece_pos(kind: str, i: int, j: int):
    if kind == "h":
        return (j * CELL + CELL / 2, i * CELL, WALL_HEIGHT / 2)
    return (j * CELL, i * CELL + CELL / 2, WALL_HEIGHT / 2)


def build_xml() -> str:
    half = CELL / 2 + WALL_THICK / 2
    walls = []
    for kind, i, j in wall_pieces():
        x, y, z = piece_pos(kind, i, j)
        size = (half, WALL_THICK / 2, WALL_HEIGHT / 2) if kind == "h" else (WALL_THICK / 2, half, WALL_HEIGHT / 2)
        walls.append(
            f'<geom name="{kind}_{i}_{j}" type="box" size="{size[0]} {size[1]} {size[2]}" '
            f'pos="{x} {y} {z}" rgba="0.18 0.55 0.62 1"/>'
        )
    dots = "\n".join(
        f'<geom name="dot_{k}" type="sphere" size="0.011" pos="0 0 {HIDDEN_Z}" contype="0" '
        f'conaffinity="0" rgba="1 0.78 0.15 1"/>'
        for k in range(TRAIL_DOTS)
    )
    sensors, rangefinders = [], []
    for name, deg in SENSOR_ANGLES.items():
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        sensors.append(f'<site name="s_{name}" pos="{0.074 * c} {0.074 * s} 0.045" xyaxes="{-s} {c} 0 0 0 1" size="0.004"/>')
        rangefinders.append(f'<rangefinder name="r_{name}" site="s_{name}" cutoff="{CUTOFF}"/>')

    return f"""
<mujoco model="maze_robot">
  <option timestep="0.002"/>
  <visual>
    <global offwidth="960" offheight="540"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.55 0.55 0.55"/>
  </visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" width="256" height="256" rgb1="0.86 0.86 0.84" rgb2="0.78 0.78 0.76"/>
    <material name="floor" texture="floor" texrepeat="40 40" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="1.5 1.5 4" dir="0 0 -1" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="floor" friction="1.2 0.005 0.0001"/>
    <geom name="start_pad" type="cylinder" size="0.16 0.002" pos="{CELL / 2} {CELL / 2} 0.002" contype="0" conaffinity="0" rgba="0.95 0.45 0.2 1"/>
    <geom name="goal_pad" type="cylinder" size="0.16 0.002" pos="0 0 {HIDDEN_Z}" contype="0" conaffinity="0" rgba="0.25 0.8 0.35 1"/>
    {chr(10).join(walls)}
    {dots}
    <body name="robot" pos="{CELL / 2} {CELL / 2} {WHEEL_R}">
      <freejoint name="root"/>
      <geom name="chassis" type="cylinder" size="0.07 0.015" pos="0 0 0.02" mass="0.5" rgba="0.95 0.45 0.2 1"/>
      <geom name="lidar" type="cylinder" size="0.026 0.012" pos="-0.01 0 0.047" mass="0.05" rgba="0.15 0.15 0.18 1"/>
      <geom name="nose" type="box" size="0.018 0.03 0.006" pos="0.055 0 0.038" mass="0.01" contype="0" conaffinity="0" rgba="1 1 1 1"/>
      <geom name="caster_f" type="sphere" size="0.0175" pos="0.05 0 -0.0105" mass="0.02" friction="0.01 0.001 0.0001" rgba="0.3 0.3 0.3 1"/>
      <geom name="caster_b" type="sphere" size="0.0175" pos="-0.05 0 -0.0105" mass="0.02" friction="0.01 0.001 0.0001" rgba="0.3 0.3 0.3 1"/>
      {chr(10).join(sensors)}
      <body name="wheel_l" pos="0 {HALF_TRACK} 0">
        <joint name="hinge_l" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom type="cylinder" size="{WHEEL_R} 0.008" euler="90 0 0" mass="0.05" friction="1.5 0.005 0.0001" rgba="0.12 0.12 0.12 1"/>
        <geom type="box" size="0.022 0.009 0.004" mass="0.0001" contype="0" conaffinity="0" rgba="0.9 0.9 0.9 1"/>
      </body>
      <body name="wheel_r" pos="0 {-HALF_TRACK} 0">
        <joint name="hinge_r" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom type="cylinder" size="{WHEEL_R} 0.008" euler="90 0 0" mass="0.05" friction="1.5 0.005 0.0001" rgba="0.12 0.12 0.12 1"/>
        <geom type="box" size="0.022 0.009 0.004" mass="0.0001" contype="0" conaffinity="0" rgba="0.9 0.9 0.9 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="motor_l" joint="hinge_l" kv="0.05" ctrllimited="true" ctrlrange="-40 40" forcelimited="true" forcerange="-0.4 0.4"/>
    <velocity name="motor_r" joint="hinge_r" kv="0.05" ctrllimited="true" ctrlrange="-40 40" forcelimited="true" forcerange="-0.4 0.4"/>
  </actuator>
  <sensor>
    {chr(10).join(rangefinders)}
  </sensor>
</mujoco>
"""


def generate_maze(seed: int) -> set:
    """Depth-first maze with no loops. Returns the wall pieces that stay closed."""
    rng = random.Random(seed)
    walls = set(wall_pieces())
    visited = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        i, j = stack[-1]
        options = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < CELLS and 0 <= nj < CELLS and (ni, nj) not in visited:
                options.append((ni, nj))
        if not options:
            stack.pop()
            continue
        ni, nj = rng.choice(options)
        if ni != i:
            walls.discard(("h", max(i, ni), j))
        else:
            walls.discard(("v", i, max(j, nj)))
        visited.add((ni, nj))
        stack.append((ni, nj))
    return walls


class MazeRobot:
    def __init__(self, seed: int = 0):
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.robot = self.model.body("robot").id
        self.sensor_adr = {n: int(self.model.sensor_adr[self.model.sensor(f"r_{n}").id]) for n in SENSOR_ANGLES}
        self.piece_ids = {p: self.model.geom(f"{p[0]}_{p[1]}_{p[2]}").id for p in wall_pieces()}
        self.dot_ids = [self.model.geom(f"dot_{k}").id for k in range(TRAIL_DOTS)]
        self.goal_id = self.model.geom("goal_pad").id
        self.goal_xy = (0.0, 0.0)
        self.next_dot = 0
        self.last_dot_time = 0.0
        self.mode = "decide"
        self.axis_yaw = 0.0
        self.seed = seed
        self.new_maze(seed)

    def new_maze(self, seed: int) -> None:
        self.seed = seed
        closed = generate_maze(seed)
        for piece, gid in self.piece_ids.items():
            x, y, z = piece_pos(*piece)
            self.model.geom_pos[gid] = (x, y, z if piece in closed else HIDDEN_Z)
        gx, gy = (CELLS - 0.5) * CELL, (CELLS - 0.5) * CELL
        self.goal_xy = (gx, gy)
        self.model.geom_pos[self.goal_id] = (gx, gy, 0.002)
        self.restart()

    def restart(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (CELL / 2, CELL / 2, WHEEL_R)
        self.data.qpos[3:7] = (1, 0, 0, 0)
        for gid in self.dot_ids:
            self.model.geom_pos[gid] = (0, 0, HIDDEN_Z)
        self.next_dot = 0
        self.last_dot_time = 0.0
        self.mode = "decide"
        self.axis_yaw = 0.0
        mujoco.mj_forward(self.model, self.data)

    def read_sensors(self) -> dict:
        out = {}
        for name, adr in self.sensor_adr.items():
            d = float(self.data.sensordata[adr])
            out[name] = CUTOFF if d < 0 else d
        return out

    def drive(self, speed: float, turn: float) -> None:
        self.data.ctrl[0] = (speed - turn * HALF_TRACK) / WHEEL_R
        self.data.ctrl[1] = (speed + turn * HALF_TRACK) / WHEEL_R

    def yaw(self) -> float:
        w, x, y, z = self.data.qpos[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def begin_move(self) -> None:
        """Drive to the center of the next cell along the current heading."""
        idx = 0 if abs(math.cos(self.axis_yaw)) > 0.5 else 1
        sign = 1 if (math.cos(self.axis_yaw) if idx == 0 else math.sin(self.axis_yaw)) > 0 else -1
        here = self.data.qpos[idx]
        self.move_idx, self.move_sign = idx, sign
        self.move_target = (round(here / CELL - 0.5) + sign + 0.5) * CELL
        self.mode = "move"

    def decide(self) -> None:
        """At a cell center: right-hand rule. Prefer right, then straight, then left, then turn around."""
        s = self.read_sensors()
        if s["right"] > OPEN_GAP:
            turn = -math.pi / 2
        elif s["front"] > OPEN_GAP:
            turn = 0.0
        elif s["left"] > OPEN_GAP:
            turn = math.pi / 2
        else:
            turn = math.pi
        self.axis_yaw = wrap(self.axis_yaw + turn)
        if turn == 0.0:
            self.begin_move()
        else:
            self.mode = "turn"

    def control(self) -> None:
        if self.mode == "decide":
            self.drive(0.0, 0.0)
            self.decide()
        elif self.mode == "turn":
            error = wrap(self.axis_yaw - self.yaw())
            if abs(error) < 0.03:
                self.drive(0.0, 0.0)
                self.begin_move()
                return
            self.drive(0.0, math.copysign(min(TURN_RATE, 0.4 + 2.5 * abs(error)), error))
        elif self.mode == "move":
            arrived = (self.data.qpos[self.move_idx] - self.move_target) * self.move_sign >= -0.01
            if arrived:
                self.decide()
                if self.mode == "move":
                    return
                self.drive(0.0, 0.0)
                return
            s = self.read_sensors()
            # steer toward the middle of the corridor, using whichever side wall is there
            if s["right"] < OPEN_GAP:
                offset = float(np.clip(LATERAL_GAIN * (s["right"] - WALL_GAP), -0.35, 0.35))
                desired = self.axis_yaw - offset
            elif s["left"] < OPEN_GAP:
                offset = float(np.clip(LATERAL_GAIN * (s["left"] - WALL_GAP), -0.35, 0.35))
                desired = self.axis_yaw + offset
            else:
                desired = self.axis_yaw
            self.drive(SPEED, float(np.clip(HEADING_GAIN * wrap(desired - self.yaw()), -TURN_RATE, TURN_RATE)))

    def drop_trail_dot(self) -> None:
        if self.data.time - self.last_dot_time < TRAIL_EVERY:
            return
        self.last_dot_time = self.data.time
        x, y = self.data.qpos[:2]
        self.model.geom_pos[self.dot_ids[self.next_dot]] = (x, y, 0.005)
        self.next_dot = (self.next_dot + 1) % TRAIL_DOTS

    def step(self) -> None:
        self.control()
        for _ in range(int(round(CONTROL_DT / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
        self.drop_trail_dot()

    def reached_goal(self) -> bool:
        x, y = self.data.qpos[:2]
        return math.hypot(x - self.goal_xy[0], y - self.goal_xy[1]) < 0.18


def selftest(seeds: int = 6, limit_s: float = 240.0) -> None:
    """Solve several random mazes with no window and report how long each takes."""
    bot = MazeRobot()
    solved = 0
    for seed in range(seeds):
        bot.new_maze(seed)
        steps = int(limit_s / CONTROL_DT)
        for k in range(steps):
            bot.step()
            if bot.reached_goal():
                solved += 1
                print(f"maze {seed}: reached the goal in {bot.data.time:5.1f} s")
                break
        else:
            x, y = bot.data.qpos[:2]
            print(f"maze {seed}: NOT solved in {limit_s:.0f} s (robot at x={x:.2f}, y={y:.2f})")
    print(f"solved {solved}/{seeds}")


def interactive() -> None:
    bot = MazeRobot(seed=random.randrange(10_000))
    wants = {"new": False, "restart": False}

    def on_key(keycode: int) -> None:
        ch = chr(keycode).upper() if 0 < keycode < 128 else ""
        if ch == "N":
            wants["new"] = True
        elif ch == "R":
            wants["restart"] = True

    print(__doc__)
    size = CELLS * CELL
    with mujoco.viewer.launch_passive(bot.model, bot.data, key_callback=on_key) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 90, -62, size * 1.55
        viewer.cam.lookat[:] = [size / 2, size / 2, 0]
        goal_time = None
        while viewer.is_running():
            started = time.perf_counter()
            if wants["new"] or (goal_time is not None and bot.data.time - goal_time > 3.0):
                bot.new_maze(random.randrange(10_000))
                wants["new"], goal_time = False, None
            elif wants["restart"]:
                bot.restart()
                wants["restart"], goal_time = False, None
            bot.step()
            if goal_time is None and bot.reached_goal():
                goal_time = bot.data.time
                print(f"Reached the goal in {goal_time:.1f} s of simulated time. New maze in 3 s.")
            if goal_time is not None:
                bot.drive(0.0, 0.0)
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - started)))


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else interactive()
