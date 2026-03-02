## 10th-place solution (Diffeomorphic Network Part)


### Installing topometrics

1. Download the dataset using `kaggle datasets download sohier/vesuvius-metric-resources`
2. Then run the following commands:

```bash
cd vesuvius-metric-resources/topological-metrics-kaggle 
pip install -r requirements.txt
chmod +x scripts/setup_submodules.sh scripts/build_betti.sh && make build-betti
pip install -e . --no-deps --no-index --no-build-isolation -v
```


