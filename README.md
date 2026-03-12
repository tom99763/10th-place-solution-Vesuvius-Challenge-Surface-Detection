## Diffeomorphic Network Part of 10th-place solution (Vesuvius Challenge - Surface Detection)

## Setup
```bash
# Create conda environment
conda create -n vesuvius python=3.10
conda activate vesuvius
pip3 install -r requirements.txt
```

### Environment 
| Hardware Setup | GPU | CPU | RAM | Time per Fold | Total (5 folds) |
| --- | --- | --- | --- | --- | --- |
| @Tom | RTX 4090 | 12th Gen Intel(R) Core(TM) i7-12700  @ 2.10 GHz | 24GB | 8-12 hours | 1.5 days |

### Installing topometrics

1. Download the dataset using `kaggle datasets download sohier/vesuvius-metric-resources`
2. Then run the following commands:

```bash
cd vesuvius-metric-resources/topological-metrics-kaggle 
pip install -r requirements.txt
chmod +x scripts/setup_submodules.sh scripts/build_betti.sh && make build-betti
pip install -e . --no-deps --no-index --no-build-isolation -v
```

### Training 
You can change config setup in `configs/config_deform_v2.yaml`. for assigning data path, please setup in yaml file.
Since we use pre-saved npy file, you have to save data as npy file first.
```
data_path: {your data path}
data_npy_path: {your npy data path}
oof_path : {your oof data path}
```

Train diffeomorphic Network:

```bash
python train_deformnet.py 
```

## Directory Structure

```
10th-place-solution-Vesuvius-Challenge-Surface-Detection-main/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── train_deformnet.py
├── cv.py
├── official_validate.py
├── configs/
│   ├── config_ICDeformnet.yaml
│   ├── config_deform.yaml
│   ├── config_deform_v2.yaml
│   ├── config_deform_v3.yaml
│   │
│   ├── data/
│   │   └── data_deform.yaml
│   │
│   ├── losses/
│   │   └── deformnet_loss.yaml
│   │
│   ├── models/
│   │   ├── deform_dynunet.yaml
│   │   ├── deform_dynunet_v2.yaml
│   │   └── ic_deform_dynunet.yaml
│   │
│   ├── optimizers/
│   │   └── AdamW.yaml
│   │
│   └── trainer/
│       ├── composeCDLTrainer.yaml
│       ├── deformSegTrainer.yaml
│       ├── pathcleanerTrainer.yaml
│       └── refineNetTrainer.yaml
│
├── src/
│   │
│   ├── datasets/
│   │   └── scroll_dataset_deform.py
│   │
│   ├── models/
│   │   ├── custom_architecture.py
│   │   └── deformNet3d.py
│   │
│   ├── nnUNet/
│   │   └── nnunetv2/
│   │       ├── evaluation/
│   │       ├── experiment_planning/
│   │       ├── imageio/
│   │       ├── inference/
│   │       ├── preprocessing/
│   │       ├── run/
│   │       ├── training/
│   │       └── utilities/
│   │
│   ├── procs/
│   │   ├── augs.py
│   │   ├── decompose.py
│   │   ├── proc_data.py
│   │   └── proc_utils.py
│   │
│   ├── trainers/
│   │   ├── deformSegTrainer.py
│   │   ├── losses.py
│   │   └── utils.py
│   │
│   └── utils/
│       ├── build_coarse_dataset.py
│       ├── kaggle_helper.py
│       ├── logger.py
│       ├── saver.py
│       ├── summaries.py
│       └── tools.py
```

## References

- Isensee et al. **Primus: Enforcing Attention Usage for 3D Medical Image Segmentation** — [openreview.net](https://openreview.net/forum?id=YWwGmmObri)
- Lamm et al. **MemBrain v2: An end-to-end tool for the analysis of membranes in cryo-electron tomography** *(Surface Dice Loss)* — [biorxiv.org](https://www.biorxiv.org/content/10.1101/2024.01.05.574336v1.full.pdf)
- Shit et al. **Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures** — [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09904.pdf)
- MIC-DKFZ **dynamic-network-architectures** library — [github.com/MIC-DKFZ/dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)
- Wald et al. **PrimusV2** *(unmerged fork)* — [github.com/TaWald/dynamic-network-architectures](https://github.com/TaWald/dynamic-network-architectures/blob/main/dynamic_network_architectures/architectures/primus.py)
- Chen et al. **Perceptual Contrastive Generative Adversarial Network based on image warping for unsupervised image-to-image translation** — [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0893608023003684)
- Ilg et al. **FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks** — [arxiv.org/abs/1612.01925](https://arxiv.org/abs/1612.01925)

---

## Contact

For any questions, feel free to reach out: tom99763@gmail.com

