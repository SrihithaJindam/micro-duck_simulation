# MicroDuck Simulation

A walking robot duck in a physics simulator, on your laptop.

Pollen Robotics' MicroDuck stands, walks, circles, turns, sits, stands up, and does a forward roll in [MuJoCo](https://github.com/google-deepmind/mujoco), driven by its trained reinforcement-learning policies.

## Requirements

| | |
|---|---|
| CPU | Any 4-core |
| RAM | 4 GB (8 GB comfortable) |
| GPU | Not needed |
| Disk | ~1.5 GB |
| Python | **3.12** ([download](https://www.python.org/downloads/)) |
| Git | [download](https://git-scm.com/downloads) |

Tested on Windows 11.

## Install

**Windows (PowerShell)**

```powershell
git clone https://github.com/Chandukasireddy/microduck-simulation.git
cd microduck-simulation
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_assets.py
```

**macOS / Linux**

```bash
git clone https://github.com/Chandukasireddy/microduck-simulation.git
cd microduck-simulation
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_assets.py
```

## Run

```bash
python demo_microduck_walk.py
```

On macOS use `mjpython` instead of `python`. Close the window to stop.

Next time: `cd microduck-simulation`, activate (`.venv\Scripts\activate` or `source .venv/bin/activate`), then run.

## Controls

- **Rotate:** left-drag · **Zoom:** scroll
- **Push the duck:** double-click it, then Ctrl + right-drag
- **Pause:** Space · **Reset:** Backspace

## More demos

```bash
python demo_dancing_dog.py  # a Unitree Go2 robot dog dances
python demo_balance.py      # an AI learns to balance a pole (~1 min)
python demo_arm_pick.py     # a robot arm picks up a cube
python demo_microduck.py    # the duck turns its head
```

## Troubleshooting

- **"Running scripts is disabled" on activate (Windows):** run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- **Install fails:** check `python --version` shows 3.12.
- **File not found:** run `python download_assets.py` again.

## Credits

- MicroDuck robot, code, and policies: [Pollen Robotics](https://github.com/pollen-robotics/microduck_rl) (Apache 2.0; the 3D meshes are CC BY-NC-SA, non-commercial only)
- Physics: [MuJoCo](https://github.com/google-deepmind/mujoco), Google DeepMind
- Motor model: [BAM](https://github.com/Rhoban/bam), Rhoban
- Robot arm and dog: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Franka Panda, Unitree Go2)
