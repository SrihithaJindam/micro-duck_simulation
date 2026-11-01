"""Download the robot models and trained policies this project needs.

Run once from the project folder:  python download_assets.py

Everything is pinned to the exact versions this project was tested with, so a
later change in the source repos won't break the demos.
"""

import subprocess
import urllib.request
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"

# Pollen Robotics' MicroDuck robot model (meshes are CC BY-NC-SA: non-commercial use only)
MICRODUCK_REPO = "https://github.com/pollen-robotics/microduck_rl.git"
MICRODUCK_COMMIT = "bae39f5900b318f7e87fc485c30a31213cfa14eb"

# Google DeepMind's robot model collection (only the Franka Panda arm and Unitree Go2 dog are downloaded)
MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "b0655057a9df2abe39ea7fc989914ba6f248301c"

# Pollen Robotics' trained MicroDuck policies on Hugging Face (Apache 2.0)
POLICY_URL = "https://huggingface.co/pollen-robotics/microduck-policies/resolve/665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448/"
POLICY_FILES = ["velstand.onnx", "alpha_sitstand.onnx", "roulade.onnx", "manifest.json"]


def git(*args: str) -> None:
    subprocess.run(["git", *args], check=True)


def clone(repo: str, commit: str, target: Path, only_folders: list[str] | None = None) -> None:
    if target.exists():
        missing = [f for f in only_folders or [] if not (target / f).exists()]
        if missing:
            print(f"Adding {', '.join(missing)} to {target.name}...")
            git("-C", str(target), "sparse-checkout", "add", *missing)
        else:
            print(f"{target.name}: already downloaded")
        return
    print(f"Downloading {target.name}...")
    extra = ["--sparse"] if only_folders else []
    git("clone", "--filter=blob:none", "--no-checkout", *extra, repo, str(target))
    if only_folders:
        git("-C", str(target), "sparse-checkout", "set", *only_folders)
    git("-C", str(target), "-c", "advice.detachedHead=false", "checkout", commit)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    clone(MICRODUCK_REPO, MICRODUCK_COMMIT, ASSETS / "microduck_rl")
    clone(
        MENAGERIE_REPO,
        MENAGERIE_COMMIT,
        ASSETS / "mujoco_menagerie",
        only_folders=["franka_emika_panda", "unitree_go2"],
    )

    policies = ASSETS / "policies"
    policies.mkdir(exist_ok=True)
    for name in POLICY_FILES:
        target = policies / name
        if target.exists():
            print(f"{name}: already downloaded")
            continue
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(POLICY_URL + name, target)

    print("All assets are ready.")


if __name__ == "__main__":
    main()
