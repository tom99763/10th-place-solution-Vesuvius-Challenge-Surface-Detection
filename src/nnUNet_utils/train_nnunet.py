import sys
import time
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from tqdm import tqdm
from scipy import ndimage
from glob import glob
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from random import sample
import nibabel as nib
from PIL import Image, ImageSequence
import shutil
import json
import subprocess
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="fft_conv_pytorch")

sys.path.append('./nnUNet')

os.environ["nnUNet_raw"] = "./nnunet/nnUNet_raw_data_base/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "./nnunet/preprocessed"
os.environ["nnUNet_results"] = "./nnunet/nnUNet_results"

#configs
plt.ion()   # interactive mode
SPACING = [1, 1, 1]  # change if needed
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_cmd(cmd_list):
    print("\n>>> Running:", " ".join(cmd_list))
    result = subprocess.run(cmd_list, check=True)
    print(">>> Done.\n")
    return result


def main():
    run_cmd([
        "nnUNetv2_plan_experiment",
        "-d", "900",
        "-c", "3d_fullres",
        "-pl", "nnUNetPlannerResEncM",
    ])


    for i in range(5):
        run_cmd([
            sys.executable,
            "-m", "nnunetv2.run.run_training",
            "900",
            "3d_fullres",
            str(i),
            "-num_gpus", "1",
            "-p", "nnUNetResEncUNetMPlans",
            '-tr', 'nnUNetTrainerBettiMatching'
        ])


if __name__ == '__main__':
    main()









