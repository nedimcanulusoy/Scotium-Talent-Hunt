import os
import sys
import subprocess
from tqdm import tqdm


def create_venv(env_name="venv"):
    # Create virtual environment
    subprocess.call([sys.executable, "-m", "venv", env_name])

    # Activate virtual environment
    activate_script = os.path.join(env_name, "Scripts", "activate.bat") if sys.platform == "win32" else os.path.join(
        env_name, "bin", "activate")
    if os.path.isfile(activate_script):
        command = f"source {activate_script}" if sys.platform != "win32" else activate_script
        subprocess.call(command, shell=True)
    else:
        print(f"Activate script not found: {activate_script}")
        return

    # Install packages
    if os.path.isfile("requirements.txt"):
        print("\033[31mThe packages of your beautiful virtual environment are loading...\033[0m")
        with open("requirements.txt", "r") as f:
            packages = f.read().splitlines()
        with tqdm(total=len(packages), desc="Packages") as pbar:
            for package in packages:
                try:
                    subprocess.call([sys.executable, "-m", "pip", "install", package], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"Error installing package {package}: {e}")
                pbar.update(1)
        print("\033[32mVirtual environment setup complete!\033[0m")
    else:
        print("No requirements.txt file found, virtual environment setup complete.")


if __name__ == "__main__":
    create_venv()  # Create virtual environment with default name "venv"
