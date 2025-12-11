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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append('./nnUNet')


#for other version
sys.path.append('/kaggle/Vesuvius-challenge-Codebase/src/vesuvius_nnunet/nnunet')
sys.path.append('/kaggle/Vesuvius-challenge-Codebase/src/vesuvius_nnunet/batchgeneratorsv2')

os.environ["nnUNet_raw"] = "./nnunet/nnUNet_raw_data_base/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "./nnunet/preprocessed"
os.environ["nnUNet_results"] = "./nnunet/nnUNet_results"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    # run_cmd([
    #     "nnUNetv2_plan_experiment",
    #     "-d", "900",
    #     "-c", "3d_fullres",
    #     "-pl", "nnUNetPlannerResEncM",
    #     #"-gpu_memory_target", "20",
    #     #"-overwrite_plans_name", "nnUNetResEncUNetMPlans"

    # ])


    for i in range(5):
        run_cmd([
            
            "nnUNetv2_train",
            "900",
            "3d_fullres",
            str(i),
            "-p", "nnUNetResEncUNetMPlans_30G",
            "-tr" , "RSNA2025TrainerSwinUNETR"
        ])


if __name__ == '__main__':
    main()









