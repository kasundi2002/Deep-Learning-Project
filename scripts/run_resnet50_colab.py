# scripts/run_resnet50_colab.py
import os, subprocess
from google.colab import drive

PROJECT = "/content/drive/MyDrive/SKIN_CANCER_PROJECT"

def main():
    # 1) Mount Drive and cd to project
    drive.mount('/content/drive', force_remount=True)
    os.chdir(PROJECT)

    # 2) Split (80/10/10). Run once, or keep --clean if you want to rebuild each time.
    subprocess.run([
        "python", "scripts/split_ham10000.py",
        "--train", "0.8", "--val", "0.1", "--test", "0.1",
        "--seed", "42", "--clean"
    ], check=True)

    # 3) Train ResNet50
    subprocess.run([
        "python", "members/run_resnet50.py",
        "--data", f"{PROJECT}/data",
        "--epochs", "20",
        "--warmup", "3",
        "--unfreeze", "10",
        "--batch", "32",
        "--out_dir", "runs",
        "--run_name", "m2_try1"
    ], check=True)

if __name__ == "__main__":
    main()


#Then in Colab just run:
# %cd /content/drive/MyDrive/SKIN_CANCER_PROJECT
# !python scripts/run_resnet50_colab.py
