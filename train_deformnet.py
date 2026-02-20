import numpy as np
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold
import hydra
from omegaconf import DictConfig, OmegaConf
import warnings
warnings.filterwarnings("ignore")
import os
import random
import torch
from src.datasets.scroll_dataset_deform import *
from src.trainers.deformSegTrainer import *
from tqdm import tqdm
import json
from src.models.deformNet3d import *
from src.trainers.deformSegTrainer import *
import torch.multiprocessing as mp
import wandb
mp.set_start_method("spawn", force=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


version = 'v2'
if version  == 'v1':
    config_name = "config_deform"
    module = DiffeoRefineModule
    _model = DeformDynUnet
elif version == 'v2':
    config_name = "config_deform_v2"
    module = DiffeoRefineModuleV2
    _model = DeformDynUnetV2
elif version == 'v3':
    config_name = "config_deform_v3"
    module = DiffeoRefineModuleV3
    _model = DeformNetV3
else:
    raise Exception('invalid version')


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(config_path="./configs", config_name=config_name, version_base=None)
def run(cfg: DictConfig):
    nnunet_path = Path(cfg.nnunet_path)
    with open(cfg.data_split_path, "r") as f:
        val_splits = json.load(f)

    for i in range(len(val_splits)):
        if cfg.selected_fold!='' and i!=cfg.selected_fold:
            continue
        print(f'training fold {i}')
        set_seed(cfg.seed)
        train_ids = list(map(lambda x: str(x), val_splits[i]['train']))
        val_ids = list(map(lambda x: str(x), val_splits[i]['val']))
        datamodule = TomoDataModule(cfg, train_ids, val_ids)
        model = _model(cfg)
        if cfg.petrained_ckpt_path != '':
            pl_model = module.load_from_checkpoint(
                cfg.petrained_ckpt_path,
                model=model,
                cfg=cfg
            )
            print('load pretrained ckpt...')
        else:
            pl_model = module(model, cfg)

        if wandb.run:
            wandb.finish()

        wnb_logger = WandbLogger(
            project=cfg.project_name,
            name=cfg.exp_name,
            config=OmegaConf.to_container(cfg),
            offline=False,
        )

        # callbacks
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_bias_comp_metric",
            mode="max",
            dirpath="./models",
            filename=f"{cfg.exp_name}-fold{i}"
                     + "-{epoch:02d}-{val_bias_comp_metric:.4f}",
            save_top_k=2,
            save_last=True
        )
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

        # trainer
        trainer = pl.Trainer(
            **cfg.trainer,
            logger=wnb_logger,
            callbacks=[lr_monitor, ckpt_callback],
        )
        # wnb_logger.watch(model, log="all", log_freq=20)

        # training
        trainer.fit(pl_model, datamodule=datamodule)

        if wnb_logger:
            wandb.finish()
        #trainer.validate(pl_model, datamodule=datamodule)

if __name__ == '__main__':
    run()