## kaggle_helper.py Usage

### Download a competition
```bash
python kaggle_helper.py download-competition \
    --competition vesuvius-challenge-surface-detection \
    --out data/
```

### Download a dataset
```bash
python kaggle_helper.py download-dataset \
    --dataset p4rallax/vesuvius-coarse-nnunet-baseline \
    --out nnunet_results/
```

### Upload a dataset
```bash
python kaggle_helper.py upload-dataset \
    --folder /kaggle/dataset \
    --username p4rallax \
    --dataset-name vesuvius-models-v2 \
    --notes "Add new model checkpoints"
```

## Train nnUNet + DeformNet3d

### Steup From Beginning
#### Train nnunet
```bash
cd nnUNet_utils
python build_nnunet_dataset.py
python train_nnunet.py
```

Download existed nnunet results:
* [nnunet-resencm-320-128-128]([https://example.com](https://www.kaggle.com/datasets/p4rallax/nnunet-resencm-320-128-128))

#### Train DeformNet3D
##### Build nnunet oof files for training DeformNet3D:
```bash
python generate_nnunet_soft_oof.py
```
You need to modify some stuffs: 
```python
import subprocess
import os
from pathlib import Path
import sys

os.environ["nnUNet_raw"] = "./nnunet/nnUNet_raw_data_base/nnUNet_raw" #your nnunet raw dataset path
os.environ["nnUNet_preprocessed"] = "./nnunet/preprocessed" #your processed dataset path
os.environ["nnUNet_results"] = "./nnunet/nnUNet_results" #trained nnunet results, including checkpoint, json files, ...

input_dir = Path("./nnunet/nnUNet_raw_data_base/nnUNet_raw/Dataset900_VesuviusScroll/imagesTr") #your training image folder path
output_dir = Path("./nnunet/nnUNet_results/Dataset900_VesuviusScroll/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/oof_softmax") #oof folder path

def run_cmd(cmd_list):
    print("\n>>> Running:", " ".join(cmd_list))
    result = subprocess.run(cmd_list, check=True)
    print(">>> Done.\n")
    return result

for i in range(5):
    run_cmd([
        sys.executable,
        "-m", "nnunetv2.run.run_training", #nnunet api
        "900", #nnunet dataset id
        "3d_fullres", #nnunet types (2d, 3d, region ....)
        str(i), #fold index
        "-num_gpus", "1",
        "-p", "nnUNetResEncUNetMPlans", #nnunet plans
        "--val", "--npz" #run validation, saving probability prediction as npz file
    ])
```
Then manually moving files of each fold `/oof_softmax/fold{i}` to `/oof_softmax` and delete all folder `/oof_softmax/fold{i}`. 

##### Configuration Setup 
Go to `configs/config_deform.yaml`, here are some stuffs need to setup:
* `data_path`: folder path of competition dataset
* `nnunet_path`: folder path of nnunet results
* `petrained_ckpt_path`: path of pretrained deformnet checkpoint
* `data_split_path`: path of validation split of nnunet json file, it's a file named `splits_final.json`

#### Train DeformNet3d
```bash
python train_deformnet.py
```

## Custom nnUNet Trainer

Go to `src/nnUNet/nnunetv2/training/nnUNetTrainer`, add your own trainer `xxxTrainer` by inheritting `nnUNetTrainer`, overriding 
your own functions: `_build_loss`, `train_step`, `validation_step`, `get_dataloaders`, `get_training_transforms`, `get_validation_transforms`
Refer to `src/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerSkeletonRecall.py`

Augmentation Example:
```python
class nnUNetTrainerCustom(nnUNetTrainer):
def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
    super().__init__(plans, configuration, fold, dataset_json, device)
    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(
            RandomTransform(
                GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True),
                apply_probability=0.1,
            )
        )
        transforms.append(
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.0),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5,
                    benchmark=True,
                ),
                apply_probability=0.2,
            )
        )
        transforms.append(
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)), synchronize_channels=False, p_per_channel=1
                ),
                apply_probability=0.15,
            )
        )
        transforms.append(
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.15,
            )
        )
        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0,
                )
            )
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)
```


Example of custom trainer and backbone:

```python
#set up plan for backbone
run_cmd([
    "nnUNetv2_plan_experiment",
    "-d", "900",
    "-c", "3d_fullres",
    "-pl", "nnUNetPlannerResEncM"
])

#training
for i in range(5):
    run_cmd([
        sys.executable,
        "-m", "nnunetv2.run.run_training",
        "900",
        "3d_fullres",
        str(i),
        "-num_gpus", "1",
        "-p", "nnUNetResEncUNetMPlans"
        "-tr", "xxxTrainer"
    ])
```



