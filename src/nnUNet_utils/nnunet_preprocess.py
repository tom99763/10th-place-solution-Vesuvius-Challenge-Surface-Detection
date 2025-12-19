import os
import json
import subprocess
from pathlib import Path

# ---------------------------
# nnU-Net paths 
# ---------------------------
os.environ["nnUNet_raw"] = "./nnunet/nnUNet_raw_data_base/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "./nnunet/preprocessed"
os.environ["nnUNet_results"] = "./nnunet/nnUNet_results"

DATASET_ID = "900"
CONFIG = "3d_fullres"
PLANNER = "nnUNetPlannerResEncM"
PLANS_NAME = "nnUNetResEncUNetMPlans_30G"
PATCH_SIZE = [320, 128, 128]


def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

# --------------------------------------------------
# 1) Extract fingerprint
# --------------------------------------------------
run([
    "nnUNetv2_extract_fingerprint",
    "-d", DATASET_ID,
    "-c", CONFIG,
    "-pl", PLANNER,
    "--verify_dataset_integrity"
])

# --------------------------------------------------
# 2) Generate plans
# --------------------------------------------------
run([
    "nnUNetv2_plan_experiment",
    "-d", DATASET_ID,
    "-c", CONFIG,
    "-pl", PLANNER,
    "-gpu_memory_target", "38",
    "-overwrite_plans_name", PLANS_NAME
])

# --------------------------------------------------
# 3) MODIFY plans file (THIS IS THE MAGIC PART)
# --------------------------------------------------
plans_dir = Path(os.environ["nnUNet_preprocessed"])
plans_file = list(plans_dir.rglob(f"{PLANS_NAME}.json"))[0]

print(f"\n>>> Editing plans file: {plans_file}")

with open(plans_file, "r") as f:
    plans = json.load(f)

plans["configurations"]["3d_fullres"]["patch_size"] = PATCH_SIZE

with open(plans_file, "w") as f:
    json.dump(plans, f, indent=4)

print(f">>> Patch size set to {PATCH_SIZE}")

# --------------------------------------------------
# 4) Preprocess
# --------------------------------------------------
run([
    "nnUNetv2_preprocess",
    "-d", DATASET_ID,
    "-c", CONFIG,
    "-pl", PLANS_NAME
])

print("\n✅ DONE: fingerprint → plans → patch size → preprocess")
