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
    --dataset_name vesuvius-models-v2 \
    --notes "Add new model checkpoints"
```

## Train nnUNet

```bash
pip install -r requirements.txt
pip install -e ./src/nnUnet/
```

### Basic training
```bash
cd nnUNet_utils
python build_nnunet_dataset.py
python train_nnunet.py
```

### Custom Trainer

Go to `src/nnUNet/nnunetv2/training/nnUNetTrainer`, add your own trainer `xxxTrainer` by inheritting `nnUNetTrainer`, overriding 
your own functions: `_build_loss`, `train_step`, `validation_step`, `get_dataloaders`, `get_training_transforms`, `get_validation_transforms`

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


### Vesuvius nnunet Trainer

These use an older version of nnunet and a modified version of batchgeneratorsv2.
You will have to run
```bash 
pip install -e src/vesuvius_nnunet/batchgeneratorsv2/
pip install -e src/vesuvius_nnunet/nnunet/
```
Note that our nnunet is the latest version of both packages, so you will have to reinstall both as per instructions at the start.
