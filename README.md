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
