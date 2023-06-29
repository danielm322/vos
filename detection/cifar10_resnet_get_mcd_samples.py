import os
import numpy as np
import torch
import hydra
# import torch.nn as nn
# import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
# from dropblock import DropBlock2D
from ls_ood_detect_cea.uncertainty_estimation import Hook, deeplabv3p_apply_dropout, get_dl_h_z
from TDL_helper_functions import get_ls_mcd_samples_baselines
from TDL_resnets import LitResnet
seed_everything(7)

PATH_DATASETS = "./cifar10_data"
# BATCH_SIZE = 256 if torch.cuda.is_available() else 64
BATCH_SIZE = 1
NUM_WORKERS = int(os.cpu_count() / 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_FOLDER = "./MCD_evaluation_data/cifar10/"
# Extract InD samples
EXTRACT_IND = True
# Extract OoD samples
EXTRACT_OOD = True


@hydra.main(version_base=None, config_path="configs/MCD_evaluation", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    assert 0 <= cfg.model.sn + cfg.model.half_sn <= 1
    img_size = cfg.model.image_size
    if cfg.model.normalize_inputs:
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(img_size, img_size)),
                torchvision.transforms.RandomCrop(img_size, padding=int(img_size/8)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )
    else:
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(img_size, img_size)),
                torchvision.transforms.RandomCrop(img_size, padding=int(img_size/8)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ]
        )
    if cfg.model.normalize_inputs:
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(img_size, img_size)),
                cifar10_normalization(),
            ]
        )
    else:
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(img_size, img_size)),
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
    # Load test set
    cifar10_dm.setup(stage="test")

    model = LitResnet(lr=0.05,
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
    # Version 0: Plain Resnet, input [-1, 1]
    # Version 1:  + layer5 (Conv2d, Relu Dropblock), input [-1, 1]
    # Version 2:  + layer5 (Conv2d, Relu Dropblock), input [0, 1]
    # Version 3:  FC (fc before final fc layer), input [0, 1]
    # Version 4:  FC (fc, dropout, fc layer), input [0, 1]
    # Version 5:  Spectral normalisation + FC no batch norm(fc, dropout, fc layer), input [0, 1]
    # Version 6:  Dropblock after 2n conv block no fc no dropout no SN input [0, 1]
    # Version 7:  Dropblock after 2n conv block no fc no dropout SN input [0, 1]
    # Version 8:  Dropblock after 2n conv block no fc no dropout SN last 2 layers input [0, 1]
    # Version 9:  Dropblock after 2n conv block no fc no dropout no SN leaky input [-1, 1]
    # Version 10:  Dropblock after 2n conv block no fc no dropout no SN leaky input [0, 1]
    # Version 12:  Dropblock after 2n conv block no fc no dropout no SN leaky half_avg_pool input [0, 1]
    # Version 13:  Dropblock after 2n conv block no fc no dropout no SN leaky full_avg_pool input [0, 1]
    # Version 14:  Dropblock after 2n conv block no fc no dropout SN leaky full_avg_pool input [0, 1]
    # Version 15:  Dropblock after 2n conv block no fc no dropout fullSN leaky full_avg_pool input [0, 1]
    # Version 16:  Dropblock after 2n conv block no fc no dropout fullSN leaky input [0, 1]
    # Version 17:  Dropblock after 1st conv block no fc no dropout fullSN leaky input avg_pool [0, 1]
    # Version 18:  Dropblock after 2n conv block no fc no dropout fullSN leaky avg_pool imsize64 input  [0, 1]
    model.load_from_checkpoint(f"./cifar10_logs/lightning_logs/version_{cfg.model_version}/checkpoints/epoch=29-step=4710.ckpt")
    model.to(device)
    # Split test set into valid and test sets
    from sklearn.model_selection import train_test_split
    valid_set, test_set = train_test_split(cifar10_dm.dataset_test, test_size=0.2, random_state=42)
    del cifar10_dm
    valid_data_loader = DataLoader(valid_set, batch_size=1)
    test_data_loader = DataLoader(test_set, batch_size=1)

    # Define ood set transforms
    if cfg.model.normalize_inputs:
        ood_transforms = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(),
                     torchvision.transforms.Resize(size=(img_size, img_size)),
                     cifar10_normalization()])
    else:
        ood_transforms = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(),
                     torchvision.transforms.Resize(size=(img_size, img_size)),
                     ])
    if cfg.ood_dataset == "gtsrb":
        # Load GTSRB
        ood_test_data = torchvision.datasets.GTSRB(
            './gtsrb_data/',
            split="test",
            download=True,
            transform=ood_transforms
        )
        test_size = 0.18
    # Load SVHN
    else:
        ood_test_data = torchvision.datasets.SVHN(
            './svhn_data/',
            split="test",
            download=True,
            transform=ood_transforms
        )
        test_size = 0.08
    valid_set_ood, test_set_ood = train_test_split(ood_test_data, test_size=test_size, random_state=42)
    # MNIST test set loader
    ood_test_loader = DataLoader(test_set_ood, batch_size=1, shuffle=True)

    # hooked_dropout_layer = Hook(model.dropblock)
    # hooked_dropout_layer = Hook(model.model.layer4._modules["1"].relu)
    hooked_dropout_layer = Hook(model.model.dropblock_layer)
    # Put model in evaluation mode
    model.eval()
    # Activate Dropout layers
    model.apply(deeplabv3p_apply_dropout)

    ###################################################################################################
    # Perform MCD inference and save samples
    ###################################################################################################
    # Get Monte-Carlo samples
    if EXTRACT_IND:
        ind_valid_mc_samples = get_ls_mcd_samples_baselines(
                model=model,
                data_loader=valid_data_loader,
                mcd_nro_samples=cfg.precomputed_mcd_runs,
                hook_dropout_layer=hooked_dropout_layer,
                layer_type=cfg.layer_type,
                device=device,
                architecture="resnet",
                location=cfg.model.dropblock_location
            )

        num_images_to_save = int(
            ind_valid_mc_samples.shape[0] / cfg.precomputed_mcd_runs
        )
        torch.save(
            ind_valid_mc_samples,
            f"./{SAVE_FOLDER}/cifar10_valid_{cfg.layer_type}_{num_images_to_save}_{ind_valid_mc_samples.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_samples.pt",
        )
        del valid_data_loader
        ind_test_mc_samples = get_ls_mcd_samples_baselines(
                model=model,
                data_loader=test_data_loader,
                mcd_nro_samples=cfg.precomputed_mcd_runs,
                hook_dropout_layer=hooked_dropout_layer,
                layer_type=cfg.layer_type,
                device=device,
                architecture="resnet",
                location=cfg.model.dropblock_location
            )

        num_images_to_save = int(
            ind_test_mc_samples.shape[0] / cfg.precomputed_mcd_runs
        )
        torch.save(
            ind_test_mc_samples,
            f"./{SAVE_FOLDER}/cifar10_test_{cfg.layer_type}_{num_images_to_save}_{ind_test_mc_samples.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_samples.pt",
        )
        del test_data_loader
    if EXTRACT_OOD:
        ood_test_mc_samples = get_ls_mcd_samples_baselines(
                model=model,
                data_loader=ood_test_loader,
                mcd_nro_samples=cfg.precomputed_mcd_runs,
                hook_dropout_layer=hooked_dropout_layer,
                layer_type=cfg.layer_type,
                device=device,
                architecture="resnet",
                location=cfg.model.dropblock_location
            )

        num_images_to_save = int(
            ood_test_mc_samples.shape[0] / cfg.precomputed_mcd_runs
        )
        torch.save(
            ood_test_mc_samples,
            f"./{SAVE_FOLDER}/{cfg.ood_dataset}_test_{cfg.layer_type}_{num_images_to_save}_{ood_test_mc_samples.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_samples.pt",
        )
        del ood_test_loader
    ########################################################################################
    # Calculate and save entropy
    ########################################################################################
    if EXTRACT_IND:
        # Calculate entropy for cifar10 valid set
        _, ind_valid_h_z_np = get_dl_h_z(
            ind_valid_mc_samples,
            mcd_samples_nro=cfg.precomputed_mcd_runs,
        )
        # Save entropy calculations
        np.save(
            f"./{SAVE_FOLDER}/cifar10_valid_{cfg.layer_type}_{ind_valid_h_z_np.shape[0]}_{ind_valid_h_z_np.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_h_z_samples",
            ind_valid_h_z_np,
        )
        del ind_test_mc_samples
        # Calculate entropy bdd test set
        _, ind_test_h_z_np = get_dl_h_z(
            ind_test_mc_samples,
            mcd_samples_nro=cfg.precomputed_mcd_runs,
        )
        # Save entropy calculations
        np.save(
            f"./{SAVE_FOLDER}/cifar10_test_{cfg.layer_type}_{ind_test_h_z_np.shape[0]}_{ind_test_h_z_np.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_h_z_samples",
            ind_test_h_z_np,
        )
        del ind_test_mc_samples
    if EXTRACT_OOD:
        # Calculate entropy ood test set
        _, ood_h_z_np = get_dl_h_z(
            ood_test_mc_samples,
            mcd_samples_nro=cfg.precomputed_mcd_runs,
        )
        # Save entropy calculations
        np.save(
            f"./{SAVE_FOLDER}/{cfg.ood_dataset}_ood_test_{cfg.layer_type}_{ood_h_z_np.shape[0]}_{ood_h_z_np.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_h_z_samples",
            ood_h_z_np,
        )


if __name__ == "__main__":
    main()
