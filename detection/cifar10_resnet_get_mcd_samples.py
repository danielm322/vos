import os
import numpy as np
import torch
import hydra
# import torch.nn as nn
# import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig
from pl_bolts.datamodules import CIFAR10DataModule

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
# from dropblock import DropBlock2D
from ls_ood_detect_cea.uncertainty_estimation import Hook, deeplabv3p_apply_dropout, get_dl_h_z
from TDL_mcd_helper_fns import MCDSamplesExtractor, get_input_transformations
from TDL_resnets import LitResnet
from TDL_datasets import SVHNDataModule

seed_everything(7)

# BATCH_SIZE = 256 if torch.cuda.is_available() else 64
BATCH_SIZE = 1
NUM_WORKERS = int(os.cpu_count() / 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Extract InD samples
EXTRACT_IND = True
# Extract OoD samples
EXTRACT_OOD = True


@hydra.main(version_base=None, config_path="configs/MCD_evaluation", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    assert 0 <= cfg.model.sn + cfg.model.half_sn <= 1
    assert cfg.ind_dataset in ("svhn", "cifar10")
    assert cfg.ood_dataset != cfg.ind_dataset
    assert cfg.ood_dataset in ("svhn", "cifar10", "gtsrb")
    SAVE_FOLDER = f"./MCD_evaluation_data/{cfg.ind_dataset}/"
    train_transforms, test_transforms = get_input_transformations(
        cifar10_normalize_inputs=cfg.model.cifar10_normalize_inputs,
        img_size=cfg.model.image_size,
        data_augmentations=cfg.data_augmentations
    )
    # Load InD dataset
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
            val_transform=test_transforms,
        )
    # Load test set
    ind_dm.setup(stage="test")

    model = LitResnet(lr=cfg.model.learning_rate,
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
                      dropblock_location=cfg.model.dropblock_location,
                      loss_type=cfg.model.loss_type,
                      original_architecture=cfg.model.original_architecture
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
    # Version 19:  Dropblock after 2n conv block no fc no dropout halfSN leaky avg_pool input  [0, 1] (server)
    # Version 20:  Dropblock after 2n conv block no fc no dropout halfSN leaky avg_pool imsize128 input  [0, 1] (server)
    # Version 21:  Dropblock after 2n conv block no fc no dropout halfSN leaky avg_pool imsize32 extra agumentations input [0, 1] (server)
    # Version 22:  Dropblock after 2n conv block no fc no dropout halfSN leaky avg_pool imsize32 extra agumentations crossentropy [0, 1] (server)
    # Version 23:  Dropblock after 2n conv block no fc no dropout halfSN leaky avg_pool imsize32 extra agumentations crossentropy original block (no double downsample) input [0, 1] (server)
    # Version 24:  Dropblock after 2n conv block no fc no dropout halfSN leaky avg_pool imsize32 extra agumentations nll original block (no double downsample) [0, 1] (server)
    # Version 25:  Dropblock after 2n conv block no fc no dropout halfSN leaky avg_pool imsize32 extra agumentations nll original block (no double downsample) SVHN [0, 1] (server)
    # Version 26:  Dropblock after 2n conv block no fc no dropout halfSN leaky avg_pool imsize32 no agumentations ce original block (no double downsample) Orig_arch [0, 1]
    # Version 27:  Dropblock 1 after 2n conv block no fc no dropout halfSN leaky avg_pool imsize32 no agumentations ce original block (no double downsample) Orig_arch [0, 1]
    if cfg.ind_dataset == "cifar10":
        model.load_from_checkpoint(f"./cifar10_logs/lightning_logs/version_{cfg.model_version}/checkpoints/epoch={cfg.model.epochs-1}-step={157*cfg.model.epochs}.ckpt")
    # SVHN
    else:
        model.load_from_checkpoint(
            f"./cifar10_logs/lightning_logs/version_{cfg.model_version}/checkpoints/epoch={cfg.model.epochs - 1}-step={287 * cfg.model.epochs}.ckpt")
    model.to(device)
    # Split test set into valid and test sets
    from sklearn.model_selection import train_test_split
    valid_set, test_set = train_test_split(
        ind_dm.dataset_test,
        test_size=0.2 if cfg.ind_dataset == "cifar10" else 0.1,
        random_state=42
    )
    del ind_dm
    valid_data_loader = DataLoader(valid_set, batch_size=1)
    test_data_loader = DataLoader(test_set, batch_size=1)

    # Load OoD Data
    if cfg.ood_dataset == "gtsrb":
        # Load GTSRB
        ood_test_data = torchvision.datasets.GTSRB(
            './gtsrb_data/',
            split="test",
            download=True,
            transform=test_transforms
        )
        test_size = 0.18
    # Load SVHN
    elif cfg.ood_dataset == "svhn":
        ood_test_data = torchvision.datasets.SVHN(
            './svhn_data/',
            split="test",
            download=True,
            transform=test_transforms
        )
        test_size = 0.08
    # Load Cifar10
    else:
        ood_test_data = torchvision.datasets.CIFAR10(
            "./cifar10_data",
            train=False,
            download=True,
            transform=test_transforms,
        )
        test_size = 0.2
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
    mcd_extractor = MCDSamplesExtractor(
        model=model,
        mcd_nro_samples=cfg.precomputed_mcd_runs,
        hook_dropout_layer=hooked_dropout_layer,
        layer_type=cfg.layer_type,
        device=device,
        architecture="resnet",
        location=cfg.model.dropblock_location,
        reduction_method=cfg.reduction_method,
        input_size=cfg.model.image_size,
        original_resnet_architecture=cfg.model.original_architecture
    )
    if EXTRACT_IND:
        # Extract InD valid
        ind_valid_mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(valid_data_loader)
        # Save
        num_images_to_save = int(
            ind_valid_mc_samples.shape[0] / cfg.precomputed_mcd_runs
        )
        torch.save(
            ind_valid_mc_samples,
            f"./{SAVE_FOLDER}/{cfg.ind_dataset}_valid_{cfg.layer_type}_{num_images_to_save}_{ind_valid_mc_samples.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_samples.pt",
        )
        del valid_data_loader
        # Extract InD test
        ind_test_mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(test_data_loader)
        # Save
        num_images_to_save = int(
            ind_test_mc_samples.shape[0] / cfg.precomputed_mcd_runs
        )
        torch.save(
            ind_test_mc_samples,
            f"./{SAVE_FOLDER}/{cfg.ind_dataset}_test_{cfg.layer_type}_{num_images_to_save}_{ind_test_mc_samples.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_samples.pt",
        )
        del test_data_loader
    if EXTRACT_OOD:
        # Extract OoD test
        ood_test_mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(ood_test_loader)
        # Save
        num_images_to_save = int(
            ood_test_mc_samples.shape[0] / cfg.precomputed_mcd_runs
        )
        torch.save(
            ood_test_mc_samples,
            f"./{SAVE_FOLDER}/{cfg.ood_dataset}_ood_test_{cfg.layer_type}_{num_images_to_save}_{ood_test_mc_samples.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_samples.pt",
        )
        del ood_test_loader
    ########################################################################################
    # Calculate and save entropy
    ########################################################################################
    if EXTRACT_IND:
        # Calculate entropy for InD valid set
        _, ind_valid_h_z_np = get_dl_h_z(
            ind_valid_mc_samples,
            mcd_samples_nro=cfg.precomputed_mcd_runs,
            parallel_run=cfg.parallel_entropy_calculation
        )
        # Save entropy calculations
        np.save(
            f"./{SAVE_FOLDER}/{cfg.ind_dataset}_valid_{cfg.layer_type}_{ind_valid_h_z_np.shape[0]}_{ind_valid_h_z_np.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_h_z_samples",
            ind_valid_h_z_np,
        )
        del ind_valid_mc_samples
        # Calculate entropy InD test set
        _, ind_test_h_z_np = get_dl_h_z(
            ind_test_mc_samples,
            mcd_samples_nro=cfg.precomputed_mcd_runs,
            parallel_run=cfg.parallel_entropy_calculation
        )
        # Save entropy calculations
        np.save(
            f"./{SAVE_FOLDER}/{cfg.ind_dataset}_test_{cfg.layer_type}_{ind_test_h_z_np.shape[0]}_{ind_test_h_z_np.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_h_z_samples",
            ind_test_h_z_np,
        )
        del ind_test_mc_samples
    if EXTRACT_OOD:
        # Calculate entropy ood test set
        _, ood_h_z_np = get_dl_h_z(
            ood_test_mc_samples,
            mcd_samples_nro=cfg.precomputed_mcd_runs,
            parallel_run=cfg.parallel_entropy_calculation
        )
        # Save entropy calculations
        np.save(
            f"./{SAVE_FOLDER}/{cfg.ood_dataset}_ood_test_{cfg.layer_type}_{ood_h_z_np.shape[0]}_{ood_h_z_np.shape[1]}_{cfg.precomputed_mcd_runs}_mcd_h_z_samples",
            ood_h_z_np,
        )


if __name__ == "__main__":
    main()
