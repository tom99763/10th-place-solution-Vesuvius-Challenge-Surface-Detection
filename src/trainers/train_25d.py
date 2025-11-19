# train_25d.py
import os
import sys
from pathlib import Path
import logging


import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from hydra.utils import instantiate

# -------------------------
# Make project root importable
# -------------------------
PROJECT_ROOT = '/kaggle/Vesuvius-challenge-Codebase'
sys.path.append(str(PROJECT_ROOT))

# Import your custom dataset + trainer
from src.datasets.scroll_dataset_25d import ScrollDataset25D
from src.trainers.seg25dTrainer import ScrollSegmentorTrainer25D


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
#                     HYDRA MAIN
# ---------------------------------------------------------
@hydra.main(version_base=None, config_path="../../configs", config_name="config_25d")
def main(cfg: DictConfig):

    print("\n====== CONFIG ======")
    print(OmegaConf.to_yaml(cfg))

    # -----------------------------------------------------
    # 1) Build model from Hydra config
    # -----------------------------------------------------
    logger.info("Instantiating model...")
    model = instantiate(cfg.models)
    # This resolves _target_: "monai.networks.nets.DynUNet"
    # and loads all arguments from your YAML.

    # -----------------------------------------------------
    # 2) Optimizer factory
    # -----------------------------------------------------
    logger.info("Setting up optimizer...")
    optimizer_factory = instantiate(cfg.optimizers, _partial_=True)

    # -----------------------------------------------------
    # 3) Trainer LightningModule
    # -----------------------------------------------------
    logger.info("Building ScrollSegmentorTrainer25D...")

    lit_model = ScrollSegmentorTrainer25D(
        model=model,
        optimizer_factory=optimizer_factory,
        scheduler_configs=cfg.get("schedulers", None),
        dataset_name=cfg.get("project_name", "unknown"),
        prediction_threshold=cfg.get("prediction_threshold", 0.5),
        batch_size=cfg.batch_size,
        input_size=cfg.input_size,
    )

    # -----------------------------------------------------
    # 4) Build dataset IDs + train/val split
    # -----------------------------------------------------
    data_path = Path(cfg.data_path)

    # Load list of training volume IDs
    # Example: ["scroll_001", "scroll_002", ...]
    id_list = sorted([p.stem for p in (data_path / "train_images_25d_1").glob("*.npy")])

    # Simple split — adjust as needed
    val_fraction = 0.1
    n_val = int(len(id_list) * val_fraction)

    train_ids = id_list[:-n_val]
    val_ids = id_list[-n_val:]

    # -----------------------------------------------------
    # 5) Build datasets
    # -----------------------------------------------------
    logger.info("Loading 2.5D datasets...")

    train_dataset = ScrollDataset25D(
        cfg=cfg.data,
        id_list=train_ids,
        
    )

    val_dataset = ScrollDataset25D(
        cfg=cfg.data,
        id_list=val_ids,
        
    )

    # -----------------------------------------------------
    # 6) DataLoaders (from Hydra dataloader config)
    # -----------------------------------------------------
    train_loader = instantiate(cfg.dataloader, dataset=train_dataset)
    val_loader = instantiate(cfg.dataloader, dataset=val_dataset, shuffle=False)

    # -----------------------------------------------------
    # 7) Lightning Trainer
    # -----------------------------------------------------
    logger.info("Creating Lightning Trainer and Module...")

    tb_logger = TensorBoardLogger("tb_logs", name=cfg.exp_name)

    # Instantiate the lightning module
    model = instantiate(cfg.models)
    lit_model = instantiate(cfg.trainer.lightning_module,model=model)

    # Instantiate the trainer
    trainer = instantiate(cfg.trainer.lightning_trainer, logger=tb_logger)

    # -----------------------------------------------------
    # 8) Train
    # -----------------------------------------------------
    logger.info("🔵 Starting training...")
    trainer.fit(lit_model, train_loader, val_loader)
    logger.info("🏁 Training complete.")


if __name__ == "__main__":
    main()
