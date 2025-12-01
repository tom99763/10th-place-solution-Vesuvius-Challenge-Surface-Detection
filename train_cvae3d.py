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
from src.datasets.scroll_dataset_cvae3d import *
from tqdm import tqdm
import json
from src.models.CVAE3d import *
from src.trainers.trainer_cvae3d import *
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)


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
    nnunet_path = Path(cfg.nnunet_path)
    with open("./splits_final.json", "r") as f:
        val_splits = json.load(f)

    for i in range(1, len(val_splits)):
        set_seed(cfg.seed)
        train_ids = val_splits[i]['train']
        val_ids = val_splits[i]['val']
        datamodule = TomoDataModule(cfg, train_ids, val_ids)
        encoder = ProgressiveEncoder3D(cfg)
        generator = ProgressiveGenerator3D(cfg)
        if cfg.petrained_ckpt_path != '':
            pl_model = ProgressiveVAETrainer.load_from_checkpoint(
                cfg.petrained_ckpt_path,
                encoder = encoder,
                generator = generator,
                cfg=cfg
            )
            print('load pretrained ckpt...')
        else:
            pl_model = ProgressiveVAETrainer(encoder, generator, cfg)
        # wnb_logger = WandbLogger(
        #     project=cfg.project_name,
        #     name=cfg.exp_name,
        #     config=OmegaConf.to_container(cfg),
        #     offline=False,
        # )

        # callbacks
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_comp_metric",
            mode="max",
            dirpath="./models",
            filename=f"{cfg.exp_name}-fold{i}-beta-{cfg.beta_kl}"
                     + "-{epoch:02d}-{val_comp_metric:.4f}",
            save_top_k=1
        )
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

        # trainer
        trainer = pl.Trainer(
            **cfg.trainer,
            #logger=wnb_logger,
            callbacks=[lr_monitor, ckpt_callback],
        )
        #wnb_logger.watch(model, log="all", log_freq=20)

        # training
        trainer.fit(pl_model, datamodule=datamodule)
        #trainer.validate(pl_model, datamodule=datamodule)

if __name__ == '__main__':
    run()