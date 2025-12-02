import json
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    path = Path("./nnunet/nnUNet_results/Dataset900_VesuviusScroll/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/oof/15307632.npz")
    x = np.load(path)
    x = x['probabilities']
    a=0