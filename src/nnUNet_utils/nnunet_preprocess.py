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

sys.path.append('/kaggle/Vesuvius-challenge-Codebase/src/nnUNet')
os.environ["nnUNet_raw"] = "/kaggle/Vesuvius-challenge-Codebase/nnunet/nnUNet_raw_data_base/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/kaggle/Vesuvius-challenge-Codebase/nnunet/preprocessed"
os.environ["nnUNet_results"] = "/kaggle/Vesuvius-challenge-Codebase/nnunet/nnUNet_results"



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
        
    #     "nnUNetv2_extract_fingerprint",
    #     "-d","900",
    #     "-c","3d_fullres",
    #     "-pl", "nnUNetPlannerResEncM",
    #     "--verify_dataset_integrity"
        
    # ])



    # run_cmd([
        
    #     "nnUNetv2_plan_and_preprocess",
    #     "-d","900",
    #     "-c","3d_fullres",
    #     "-pl", "nnUNetPlannerResEncM",
    #     "--verify_dataset_integrity"
        
    # ])
    # # Only generate plans
    #    run_cmd([
        
    #     "nnUNetv2_plan_experiment",
    #     "-d","900",
    #     "-c","3d_fullres",
    #     "-pl", "nnUNetPlannerResEncM",
    #     "-gpu_memory_target" , "38",
    #     "-overwrite_plans_name", "nnUNetResEncUNetMPlans_30G"
        
    # ])

    run_cmd([
        
        "nnUNetv2_preprocess",
        "-d","900",
        "-c","3d_fullres",
        "-pl", "nnUNetResEncUNetMPlans_30G",
        
        
    ])


if __name__ == '__main__':
    main()









