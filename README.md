## Diffeomorphic Network Part of 10th-place solution (Vesuvius Challenge - Surface Detection)

## Install Dependencies
```bash
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


