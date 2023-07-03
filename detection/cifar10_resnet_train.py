import os
import pandas as pd
import seaborn as sn
import torch
import hydra
from IPython.core.display import display
from omegaconf import DictConfig
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from TDL_resnets import LitResnet
from TDL_mcd_helper_fns import get_input_transformations
from TDL_datasets import SVHNDataModule

seed_everything(7)
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)


@hydra.main(version_base=None, config_path="configs/MCD_evaluation/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    assert 0 <= cfg.model.sn + cfg.model.half_sn <= 1
    assert cfg.ind_dataset in ("cifar10", "svhn")
    train_transforms, test_transforms = get_input_transformations(
        cifar10_normalize_inputs=cfg.model.cifar10_normalize_inputs,
        img_size=cfg.model.image_size,
        extra_augmentations=cfg.extra_data_augmentations
    )
    if cfg.ind_dataset == "cifar10":
        PATH_DATASETS = "./cifar10_data"
        ind_dm = CIFAR10DataModule(
            data_dir=PATH_DATASETS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=test_transforms,
        )
    # SVHN dataset
    else:
        PATH_DATASETS = "./svhn_data"
        ind_dm = SVHNDataModule(
            data_dir=PATH_DATASETS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            train_transform=train_transforms,
            test_transform=test_transforms,
        )

    model = LitResnet(lr=0.05,
                      num_classes=10,
                      spectral_normalization=cfg.model.sn,
                      fifth_conv_layer=cfg.model.fifth_conv_layer,
                      extra_fc_layer=cfg.model.extra_fc,
                      dropout=cfg.model.dropout,
                      dropblock=cfg.model.dropblock,
                      half_sn=cfg.model.half_sn,
                      activation=cfg.model.activation,
                      dropblock_prob=cfg.model.dropblock_prob,
                      dropblock_size=cfg.model.dropblock_size,
                      dropout_prob=cfg.model.dropout_prob,
                      avg_pool=cfg.model.avg_pool,
                      dropblock_location=cfg.model.dropblock_location
                      )

    trainer = Trainer(
        max_epochs=45,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir="cifar10_logs/"),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, ind_dm)
    trainer.test(model, datamodule=ind_dm)

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    display(metrics.dropna(axis=1, how="all").head())
    sn.relplot(data=metrics, kind="line")


if __name__ == "__main__":
    main()
