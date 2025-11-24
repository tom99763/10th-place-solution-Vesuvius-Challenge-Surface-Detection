import numpy as np
import pandas as pd
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


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(config_path="./configs", config_name="config_deform", version_base=None)
def run(cfg: DictConfig):
    meta_data = pd.read_csv(f'{cfg.data_path}/train.csv')

    #validation
    #load splits_final.json

    # #assign fold
    # meta_data['fold'] = -1
    # skf = StratifiedKFold(n_splits=cfg.n_splits)
    # for i, (train_idx, val_idx) in enumerate(skf.split(meta_data)):
    #
    #     set_seed(cfg.seed)
    #
    #     datamodule = SequenceDataModule(meta_data, train_idx, val_idx, cfg)
    #
    #     model = instantiate(cfg.models)
    #
    #     pl_model = SequenceTrainer(model, cfg)
    #
    #     wnb_logger = WandbLogger(
    #         project=cfg.project_name,
    #         name=f"{cfg.exp_name}-{cfg.wind_size}-{cfg.wind_step}-aux-fold{i}",
    #         config=OmegaConf.to_container(cfg),
    #         offline=False,
    #     )
    #
    #     # callbacks
    #     ckpt_callback = pl.callbacks.ModelCheckpoint(
    #         monitor="val_f1",
    #         mode="max",
    #         dirpath="./models",
    #         filename=f"{cfg.exp_name}-fold{i}"
    #                  + "-{epoch:02d}-{val_loss:.4f}-{val_f1:.4f}",
    #         save_top_k=1
    #     )
    #     lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    #
    #     # trainer
    #     trainer = pl.Trainer(
    #         **cfg.trainer,
    #         logger=wnb_logger,
    #         callbacks=[lr_monitor, ckpt_callback],
    #     )
    #     wnb_logger.watch(model, log="all", log_freq=20)
    #
    #     # training
    #     trainer.fit(pl_model, datamodule=datamodule)
    #     #trainer.validate(pl_model, datamodule=datamodule)


if __name__ == '__main__':
    run()