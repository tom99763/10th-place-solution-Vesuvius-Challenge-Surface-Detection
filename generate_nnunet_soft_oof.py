import subprocess
import os
from pathlib import Path
import sys

os.environ["nnUNet_raw"] = "./nnunet/nnUNet_raw_data_base/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "./nnunet/preprocessed"
os.environ["nnUNet_results"] = "./nnunet/nnUNet_results"

input_dir = Path("./nnunet/nnUNet_raw_data_base/nnUNet_raw/Dataset900_VesuviusScroll/imagesTr")
output_dir = Path("./nnunet/nnUNet_results/Dataset900_VesuviusScroll/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/oof_softmax")

# for i in range(5):
#     if os.path.exists(output_dir/f'fold{i}'):
#         os.makedirs(output_dir/f'fold{i}')
# cmd = [
#     "nnUNetv2_predict",
#     "-i", str(input_dir),
#     "-o", str(output_dir/f'fold{i}'),
#     "-d", "900",
#     "-c", "3d_fullres",
#     "-p", "nnUNetResEncUNetMPlans",
#     "--save_probability",
#     "--disable_tta"
# ]

def run_cmd(cmd_list):
    print("\n>>> Running:", " ".join(cmd_list))
    result = subprocess.run(cmd_list, check=True)
    print(">>> Done.\n")
    return result

for i in range(5):
    run_cmd([
        sys.executable,
        "-m", "nnunetv2.run.run_training",
        "900",
        "3d_fullres",
        str(i),
        "-num_gpus", "1",
        "-p", "nnUNetResEncUNetMPlans",
        "--val", "--npz"
    ])