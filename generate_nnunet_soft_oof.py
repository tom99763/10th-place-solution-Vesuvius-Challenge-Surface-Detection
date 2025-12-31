import subprocess
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["nnUNet_raw"] = "./nnunet/nnUNet_raw_data_base/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "./nnunet/preprocessed"
os.environ["nnUNet_results"] = "./nnunet/nnUNet_results"

input_dir = Path("./nnunet/nnUNet_raw_data_base/nnUNet_raw/Dataset900_VesuviusScroll/imagesTr")
output_dir = Path("./nnunet/nnUNet_results/Dataset900_VesuviusScroll/nnUNetTrainer__nnUNetResEncUNetMPlans_30G__3d_fullres/oof")

# cmd = [
#     "nnUNetv2_predict",
#     "-i", str(input_dir),
#     "-o", str(output_dir),
#     "-d", "900",
#     "-c", "3d_fullres",
#     "-p", "nnUNetResEncUNetMPlans_30G",
#     "--save_probabilities",
#     "--disable_tta"
# ]

def run_cmd(cmd_list):
    print("\n>>> Running:", " ".join(cmd_list))
    result = subprocess.run(cmd_list, check=True)
    print(">>> Done.\n")
    return result

#run_cmd(cmd)

for i in range(1):
    run_cmd([
        sys.executable,
        "-m", "nnunetv2.run.run_training",
        "900",
        "3d_fullres",
        str(i),
        "-num_gpus", "1",
        "-p", "nnUNetResEncUNetMPlans",
        "--val" #, "--npz"
    ])