import os
import pandas as pd
import seaborn as sn
import torch
import hydra
# import torch.nn as nn
import torchvision
from IPython.core.display import display
from omegaconf import DictConfig
# from dropblock import DropBlock2D
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
# from torch.optim.swa_utils import AveragedModel, update_bn
from TDL_resnets import LitResnet

seed_everything(7)
PATH_DATASETS = "./cifar10_data"
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)


@hydra.main(version_base=None, config_path="configs/MCD_evaluation/model/", config_name="model.yaml")
def main(cfg: DictConfig) -> None:
    assert 0 <= cfg.sn + cfg.half_sn <= 1
    if cfg.normalize_inputs:
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )
    else:
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ]
        )
    if cfg.normalize_inputs:
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )
    else:
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

    cifar10_dm = CIFAR10DataModule(
        data_dir=PATH_DATASETS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    model = LitResnet(lr=0.05,
                      spectral_normalization=cfg.sn,
                      fifth_conv_layer=cfg.fifth_conv_layer,
                      extra_fc_layer=cfg.extra_fc,
                      dropout=cfg.dropout,
                      dropblock=cfg.dropblock,
                      half_sn=cfg.half_sn,
                      activation=cfg.activation,
                      dropblock_prob=cfg.dropblock_prob,
                      dropblock_size=cfg.dropblock_size,
                      dropout_prob=cfg.dropout_prob,
                      avg_pool=cfg.avg_pool,
                      dropblock_location=cfg.dropblock_location
                      )

    trainer = Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir="cifar10_logs/"),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    display(metrics.dropna(axis=1, how="all").head())
    sn.relplot(data=metrics, kind="line")


if __name__ == "__main__":
    main()
