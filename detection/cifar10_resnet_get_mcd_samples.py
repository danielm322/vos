import os
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
# from dropblock import DropBlock2D
from ls_ood_detect_cea.uncertainty_estimation import Hook, deeplabv3p_apply_dropout, get_dl_h_z
from TDL_helper_functions import get_ls_mcd_samples_baselines
from TDL_resnets import LitResnetFC
seed_everything(7)

PATH_DATASETS = "./cifar10_data"
# BATCH_SIZE = 256 if torch.cuda.is_available() else 64
BATCH_SIZE = 1
NUM_WORKERS = int(os.cpu_count() / 2)
DROPBLOCK_PROB = 0.3
MCD_RUNS = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_FOLDER = "./MCD_evaluation_data/cifar10/"
LAYER_TYPE = "FC"
OOD_DATASET = "gtsrb"
# Extract InD samples
EXTRACT_IND = True
# Extract OoD samples
EXTRACT_OOD = True


# class LitResnetMCD(LitResnet):
#     def __init__(self):
#         super().__init__(LitResnet)
#         self.model.dropblock = DropBlock2D(DROPBLOCK_PROB, block_size=1)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # See note [TorchScript super()]
#         x = self.model.conv1(x)
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x)
#
#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         x = self.model.layer4(x)
#         x = self.model.dropblock(x)
#         x = self.model.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.model.fc(x)
#         return F.log_softmax(x, dim=1)


def main():
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            # cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # cifar10_normalization(),
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

    model = LitResnetFC()
    # Version 0: Plain Resnet, input [-1, 1]
    # Version 1: Resnet + layer5 (Conv2d, Relu Dropblock), input [-1, 1]
    # Version 2: Resnet + layer5 (Conv2d, Relu Dropblock), input [0, 1]
    # Version 3: Resnet FC (fc before final fc layer), input [0, 1]
    # Version 4: Resnet FC (fc, dropout, fc layer), input [0, 1]
    # Version 5: Resnet Spectral normalisation + FC (fc, dropout, fc layer), input [0, 1]
    model.load_from_checkpoint("./cifar10_logs/lightning_logs/version_5/checkpoints/epoch=29-step=4710.ckpt")
    model.to(device)
    # Split test set into valid and test sets
    from sklearn.model_selection import train_test_split
    valid_set, test_set = train_test_split(cifar10_dm.dataset_test, test_size=0.2, random_state=42)
    valid_data_loader = DataLoader(valid_set, batch_size=1)
    test_data_loader = DataLoader(test_set, batch_size=1)

    if OOD_DATASET == "gtsrb":
        # Load GTSRB
        ood_test_data = torchvision.datasets.GTSRB(
            './gtsrb_data/',
            split="test",
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Resize(size=(32, 32))
                 ])
                 # cifar10_normalization()])
        )
        test_size = 0.2
    # Load SVHN
    else:
        ood_test_data = torchvision.datasets.SVHN(
            './svhn_data/',
            split="test",
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Resize(size=(32, 32))
                 ])
            # cifar10_normalization()])
        )
        test_size = 0.08
    valid_set_ood, test_set_ood = train_test_split(ood_test_data, test_size=test_size, random_state=42)
    # MNIST test set loader
    ood_test_loader = DataLoader(test_set_ood, batch_size=1, shuffle=True)

    # hooked_dropout_layer = Hook(model.dropblock)
    # hooked_dropout_layer = Hook(model.model.layer4._modules["1"].relu)
    hooked_dropout_layer = Hook(model.model.dropout)
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
                mcd_nro_samples=MCD_RUNS,
                hook_dropout_layer=hooked_dropout_layer,
                layer_type=LAYER_TYPE,
                device=device,
                architecture="resnet"
            )

        num_images_to_save = int(
            ind_valid_mc_samples.shape[0] / MCD_RUNS
        )
        torch.save(
            ind_valid_mc_samples,
            f"./{SAVE_FOLDER}/cifar10_valid_{LAYER_TYPE}_{num_images_to_save}_{ind_valid_mc_samples.shape[1]}_{MCD_RUNS}_mcd_samples.pt",
        )

        ind_test_mc_samples = get_ls_mcd_samples_baselines(
                model=model,
                data_loader=test_data_loader,
                mcd_nro_samples=MCD_RUNS,
                hook_dropout_layer=hooked_dropout_layer,
                layer_type=LAYER_TYPE,
                device=device,
                architecture="resnet"
            )

        num_images_to_save = int(
            ind_test_mc_samples.shape[0] / MCD_RUNS
        )
        torch.save(
            ind_test_mc_samples,
            f"./{SAVE_FOLDER}/cifar10_test_{LAYER_TYPE}_{num_images_to_save}_{ind_test_mc_samples.shape[1]}_{MCD_RUNS}_mcd_samples.pt",
        )
    if EXTRACT_OOD:
        ood_test_mc_samples = get_ls_mcd_samples_baselines(
                model=model,
                data_loader=ood_test_loader,
                mcd_nro_samples=MCD_RUNS,
                hook_dropout_layer=hooked_dropout_layer,
                layer_type=LAYER_TYPE,
                device=device,
                architecture="resnet"
            )

        num_images_to_save = int(
            ood_test_mc_samples.shape[0] / MCD_RUNS
        )
        torch.save(
            ood_test_mc_samples,
            f"./{SAVE_FOLDER}/{OOD_DATASET}_test_{LAYER_TYPE}_{num_images_to_save}_{ood_test_mc_samples.shape[1]}_{MCD_RUNS}_mcd_samples.pt",
        )

    ########################################################################################
    # Calculate and save entropy
    ########################################################################################
    if EXTRACT_IND:
        # Calculate entropy for cifar10 valid set
        _, ind_valid_h_z_np = get_dl_h_z(
            ind_valid_mc_samples,
            mcd_samples_nro=MCD_RUNS,
        )
        # Save entropy calculations
        np.save(
            f"./{SAVE_FOLDER}/cifar10_valid_{LAYER_TYPE}_{ind_valid_h_z_np.shape[0]}_{ind_valid_h_z_np.shape[1]}_{MCD_RUNS}_mcd_h_z_samples",
            ind_valid_h_z_np,
        )
        # Calculate entropy bdd test set
        _, ind_test_h_z_np = get_dl_h_z(
            ind_test_mc_samples,
            mcd_samples_nro=MCD_RUNS,
        )
        # Save entropy calculations
        np.save(
            f"./{SAVE_FOLDER}/cifar10_test_{LAYER_TYPE}_{ind_test_h_z_np.shape[0]}_{ind_test_h_z_np.shape[1]}_{MCD_RUNS}_mcd_h_z_samples",
            ind_test_h_z_np,
        )
    if EXTRACT_OOD:
        # Calculate entropy ood test set
        _, ood_h_z_np = get_dl_h_z(
            ood_test_mc_samples,
            mcd_samples_nro=MCD_RUNS,
        )
        # Save entropy calculations
        np.save(
            f"./{SAVE_FOLDER}/{OOD_DATASET}_ood_test_{LAYER_TYPE}_{ood_h_z_np.shape[0]}_{ood_h_z_np.shape[1]}_{MCD_RUNS}_mcd_h_z_samples",
            ood_h_z_np,
        )


if __name__ == "__main__":
    main()
