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

## Train nnUNet

### Basic training
```bash
cd nnUNet_utils
python build_nnunet_dataset.py
python train_nnunet.py
```

### Custom Trainer

Go to `src/nnUNet/nnunetv2/training/nnUNetTrainer`, add your own trainer `xxxTrainer` by inheritting `src/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`, overriding 
your own functions: `_build_loss`, `train_step`, `validation_step`, `get_dataloaders`, `get_training_transforms`, `get_validation_transforms`
this can refer to `src/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerSkeletonRecall.py`

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

