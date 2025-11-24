import subprocess
import os

os.environ["nnUNet_raw"] = "./nnunet/nnUNet_raw_data_base/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "./nnunet/preprocessed"
os.environ["nnUNet_results"] = "./nnunet/nnUNet_results"



# command as a list of arguments
cmd = [
    "nnUNetv2_predict",
    "-i", "./nnunet/nnUNet_raw_data_base/nnUNet_raw/Dataset900_VesuviusScroll/imagesTr",
    "-o", "./nnunet/nnUNet_results/Dataset900_VesuviusScroll/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/oof_softmax",
    "-d", "900",
    "-c", "3d_fullres",
    "-p", "nnUNetResEncUNetMPlans",
    "--save_probabilities"
]

# run the command
result = subprocess.run(cmd, check=True)