# train_25d.py
import os
import sys
from pathlib import Path
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from hydra.utils import instantiate

# -------------------------
# Make project root importable
# -------------------------
PROJECT_ROOT = 'C:/Users/tom99/PycharmProjects/Vesuvius-challenge-Codebase'
sys.path.append(str(PROJECT_ROOT))

# Import your custom dataset + trainer
from src.datasets.scroll_dataset_25d import *
from src.trainers.seg25dTrainer import *
import torch
from lightning.pytorch.loggers import WandbLogger
torch.set_float32_matmul_precision('medium')

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
#                     HYDRA MAIN
# ---------------------------------------------------------
@hydra.main(version_base=None, config_path="../../configs", config_name="config_25d")
def main(cfg: DictConfig):

    print("\n====== CONFIG ======")
    print(OmegaConf.to_yaml(cfg))

    wnb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.exp_name,
        config=OmegaConf.to_container(cfg),
        offline=cfg.offline,
    )
    logger.info("Building ScrollSegmentorTrainer25D...")

    data_path = Path(cfg.data_path)
    id_list = list((data_path / "train_images_25d_1").glob("*/*.npy"))
    n_val = int(len(id_list) * cfg.val_ratio)

    train_ids = id_list[:-n_val]
    val_ids = id_list[-n_val:]
    logger.info("Loading 2.5D datasets...")

    datamodule = ScrollDataModule25D(
        cfg=cfg,
        train_ids = train_ids,
        val_ids = val_ids
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_dice"
        , mode="max"
        , dirpath="./models"
        , filename=f'{cfg.exp_name}' + '-{epoch:02d}-{val_loss:.4f}-{"val_dice:.4f}'+ \
                   f"fold_id={cfg.fold_id}"
        , save_top_k=1
        , save_last=True
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # -----------------------------------------------------
    # 7) Lightning Trainer
    # -----------------------------------------------------
    logger.info("Creating Lightning Trainer and Module...")
    model = instantiate(cfg.models)
    lit_model = instantiate(cfg.trainer.lightning_module)(model=model)

    trainer = hydra.utils.instantiate(cfg.trainer.lightning_trainer)
    trainer_additional_kwargs = {
        "logger": wnb_logger,
        "callbacks": [lr_monitor, ckpt_callback],
        "devices": cfg.devices
    }
    trainer = trainer(**trainer_additional_kwargs)

    wnb_logger.watch(model, log="all", log_freq=20)
    logger.info("🔵 Starting training...")
    trainer.fit(lit_model, datamodule=datamodule)
    logger.info("🏁 Training complete.")
    #trainer.validate(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
