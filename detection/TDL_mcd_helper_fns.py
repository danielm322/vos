from typing import Tuple
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import numpy as np
import torch
import torchvision
from ls_ood_detect_cea import DetectorKDE, get_hz_scores
from torch.nn.functional import avg_pool2d
from dropblock import DropBlock2D
from ls_ood_detect_cea.uncertainty_estimation import Hook
from tqdm import tqdm

# Ugly but fast way to test to hook the backbone: get raw output, apply dropblock
# inside the following function
dropblock_ext = DropBlock2D(drop_prob=0.4, block_size=1)
dropout_ext = torch.nn.Dropout(p=0.5)


# Get latent space Monte Carlo Dropout samples
def get_ls_mcd_samples_rcnn(model: torch.nn.Module,
                            data_loader: torch.utils.data.dataloader.DataLoader,
                            mcd_nro_samples: int,
                            hook_dropout_layer: Hook,
                            layer_type: str) -> torch.tensor:
    """
     Get Monte-Carlo samples from any torch model Dropout or Dropblock Layer
        THIS FUNCTION SHOULD BE ADDED INTO THE LS OOD DETECTION LIBRARY
     :param model: Torch model
     :type model: torch.nn.Module
     :param data_loader: Input samples (torch) DataLoader
     :type data_loader: DataLoader
     :param mcd_nro_samples: Number of Monte-Carlo Samples
     :type mcd_nro_samples: int
     :param hook_dropout_layer: Hook at the Dropout Layer from the Neural Network Module
     :type hook_dropout_layer: Hook
     :param layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected) or Conv (Convolutional)
     :type: str
     :return: Monte-Carlo Dropout samples for the input dataloader
     :rtype: Tensor
     """
    assert layer_type in ("FC", "Conv", "RPN", "backbone"), "Layer type must be either 'FC', 'RPN' or 'Conv'"
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Extracting MCD samples") as pbar:
            dl_imgs_latent_mcd_samples = []
            for i, image in enumerate(data_loader):
                img_mcd_samples = []
                for s in range(mcd_nro_samples):
                    pred_img = model(image)
                    # pred = torch.argmax(pred_img, dim=1)
                    latent_mcd_sample = hook_dropout_layer.output
                    if layer_type == "Conv":
                        # Get image HxW mean:
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                        # Remove useless dimensions:
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=3)
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                    elif layer_type == "RPN":
                        latent_mcd_sample = model.model.proposal_generator.rpn_head.rpn_intermediate_output
                        for i in range(len(latent_mcd_sample)):
                            latent_mcd_sample[i] = torch.mean(latent_mcd_sample[i], dim=2, keepdim=True)
                            latent_mcd_sample[i] = torch.mean(latent_mcd_sample[i], dim=3, keepdim=True)
                            # Remove useless dimensions:
                            latent_mcd_sample[i] = torch.squeeze(latent_mcd_sample[i])
                        latent_mcd_sample = torch.cat(latent_mcd_sample, dim=0)
                    elif layer_type == "backbone":
                        # Apply dropblock
                        for k, v in latent_mcd_sample.items():
                            latent_mcd_sample[k] = dropblock_ext(v)
                            # Get image HxW mean:
                            latent_mcd_sample[k] = torch.mean(latent_mcd_sample[k], dim=2, keepdim=True)
                            latent_mcd_sample[k] = torch.mean(latent_mcd_sample[k], dim=3, keepdim=True)
                            # Remove useless dimensions:
                            latent_mcd_sample[k] = torch.squeeze(latent_mcd_sample[k])
                        latent_mcd_sample = torch.cat(list(latent_mcd_sample.values()), dim=0)
                    else:
                        # Aggregate the second dimension (dim 1) to keep the proposed boxes dimension
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=1)

                    img_mcd_samples.append(latent_mcd_sample)
                if layer_type == "Conv":
                    img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
                else:
                    img_mcd_samples_t = torch.stack(img_mcd_samples, dim=0)
                dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)
                # Update progress bar
                pbar.update(1)
            dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)

    return dl_imgs_latent_mcd_samples_t


def get_ls_mcd_samples_baselines(model: torch.nn.Module,
                                 data_loader: torch.utils.data.dataloader.DataLoader,
                                 mcd_nro_samples: int,
                                 hook_dropout_layer: Hook,
                                 layer_type: str,
                                 device: str,
                                 architecture: str,
                                 location: int,
                                 reduction_method: str,
                                 input_size: int) -> torch.tensor:
    """
     Get Monte-Carlo samples from any torch model Dropout or Dropblock Layer
        THIS FUNCTION SHOULD BE ADDED INTO THE LS OOD DETECTION LIBRARY
     :param model: Torch model
     :type model: torch.nn.Module
     :param data_loader: Input samples (torch) DataLoader
     :type data_loader: DataLoader
     :param mcd_nro_samples: Number of Monte-Carlo Samples
     :type mcd_nro_samples: int
     :param hook_dropout_layer: Hook at the Dropout Layer from the Neural Network Module
     :type hook_dropout_layer: Hook
     :param layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected) or Conv (Convolutional)
     :type: str
     :param architecture: The model architecture: either small or resnet
     :param location: Location of the hook. This can be useful to select different latent sample catching layers
     :return: Monte-Carlo Dropout samples for the input dataloader
     :rtype: Tensor
     """
    assert layer_type in ("FC", "Conv"), "Layer type must be either 'FC' or 'Conv'"
    assert architecture in ("small", "resnet"), "Only 'small' or 'resnet' are supported"
    assert input_size in (32, 64, 128)
    if architecture == "resnet" and location in (1, 2):
        assert reduction_method in ("mean", "avgpool"), "Only mean and avg pool reduction method supported for resnet"
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Extracting MCD samples") as pbar:
            dl_imgs_latent_mcd_samples = []
            for i, (image, label) in enumerate(data_loader):
                # image = image.view(1, 1, 28, 28).to(device)
                image = image.to(device)
                img_mcd_samples = []
                for s in range(mcd_nro_samples):
                    pred_img = model(image)
                    # pred = torch.argmax(pred_img, dim=1)
                    latent_mcd_sample = hook_dropout_layer.output
                    if layer_type == "Conv":
                        if architecture == "small":
                            # Get image HxW mean:
                            latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                            # latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                            # Remove useless dimensions:
                            # latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=3)
                            latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                            latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                        # Resnet 18
                        else:
                            # latent_mcd_sample = dropblock_ext(latent_mcd_sample)
                            # For 2nd conv layer block of resnet 18:
                            if location == 2:
                                # To conserve the most info, while also aggregating: let us reshape then average
                                if input_size == 32:
                                    assert latent_mcd_sample.shape == torch.Size([1, 128, 8, 8])
                                    if reduction_method == "mean":
                                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                        latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                    # Avg pool
                                    else:
                                        # Perform average pooling over latent representations
                                        # For input of size 32
                                        latent_mcd_sample = avg_pool2d(latent_mcd_sample, kernel_size=4, stride=3, padding=1)
                                # Input size 64
                                elif input_size == 64:
                                    assert latent_mcd_sample.shape == torch.Size([1, 128, 16, 16])
                                    if reduction_method == "mean":
                                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                        latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 4, -1)
                                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                        latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                    else:
                                        # For input of size 64
                                        latent_mcd_sample = avg_pool2d(latent_mcd_sample, kernel_size=8, stride=6, padding=2)
                                # Input size 128
                                else:
                                    assert latent_mcd_sample.shape == torch.Size([1, 128, 32, 32])
                                    if reduction_method == "mean":
                                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                        latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 4, -1)
                                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                        latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                    else:
                                        # For input of size 64
                                        latent_mcd_sample = avg_pool2d(latent_mcd_sample, kernel_size=16, stride=12, padding=4)

                                latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                            elif location == 1:
                                assert latent_mcd_sample.shape == torch.Size([1, 64, 32, 32])
                                # latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 4, -1)
                                if reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = latent_mcd_sample.reshape(1, 64, 16, -1)
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                # Avg pool
                                else:
                                    latent_mcd_sample = avg_pool2d(latent_mcd_sample, kernel_size=4, stride=2, padding=2)
                                latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                            elif location == 3:
                                assert latent_mcd_sample.shape == torch.Size([1, 256, 2, 2])
                                # latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 4, -1)
                                latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                            else:
                                raise NotImplementedError
                                # Get image HxW mean:
                                # latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                                # latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                # # Remove useless dimensions:
                                # latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=3)
                                # latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                                # latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                    # FC
                    else:
                        # It is already a 1d tensor
                        # latent_mcd_sample = dropout_ext(latent_mcd_sample)
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample)

                    img_mcd_samples.append(latent_mcd_sample)
                if layer_type == "Conv":
                    img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
                else:
                    img_mcd_samples_t = torch.stack(img_mcd_samples, dim=0)
                dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)
                # Update progress bar
                pbar.update(1)
            dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)
    print("MCD N_samples: ", dl_imgs_latent_mcd_samples_t.shape[1])
    return dl_imgs_latent_mcd_samples_t


def fit_evaluate_KDE(h_z_ind_valid_samples: np.array,
                     h_z_ind_test_samples: np.array,
                     h_z_ood_samples: np.array,
                     normalize: bool, ) -> Tuple[np.array, np.array]:
    """
    This function fits and evaluates a KDE classifier using valid and test samples from an In Distribution set,
    compared to an OoD set. It returns evaluation metrics to be logged with MLFlow
    :param normalize: Whether to normalize the kde transformation
    :param h_z_ind_valid_samples: InD valid samples to build the KDE
    :param h_z_ind_test_samples: InD test samples to evaluate the KDE
    :param h_z_ood_samples: OoD samples
    :return:
    """
    bdd_ds_shift_detector = DetectorKDE(train_embeddings=h_z_ind_valid_samples)
    if normalize:
        scores_bdd_test = get_hz_scores(hz_detector=bdd_ds_shift_detector,
                                        samples=h_z_ind_test_samples)
        scores_ood = get_hz_scores(hz_detector=bdd_ds_shift_detector,
                                   samples=h_z_ood_samples)
    else:
        scores_bdd_test = bdd_ds_shift_detector.density.score_samples(h_z_ind_test_samples)
        scores_ood = bdd_ds_shift_detector.density.score_samples(h_z_ood_samples)

    return scores_bdd_test, scores_ood


def adjust_mlflow_results_name(data_dict: dict, technique_name: str) -> dict:
    """
    This function simply adds the name of the dimension reduciton technique at the end of the metrics names,
    In order to facilitate analysis with mlflow
    :param data_dict: Metrics dictionary
    :param technique_name: Either pca or pm (pacmap)
    :return: Dictionary with changed keys
    """
    new_dict = dict()
    for k, v in data_dict.items():
        new_dict[k + f'_{technique_name}'] = v
    return new_dict


def reduce_mcd_samples(bdd_valid_mc_samples: torch.Tensor,
                       bdd_test_mc_samples: torch.Tensor,
                       ood_test_mc_samples: torch.Tensor,
                       precomputed_mcd_runs: int,
                       n_mcd_runs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function takes precomputed Monte Carlo Dropout samples, and returns a smaller number of samples to work with
    :param bdd_valid_mc_samples:
    :param bdd_test_mc_samples:
    :param ood_test_mc_samples:
    :param precomputed_mcd_runs:
    :param n_mcd_runs:
    :return: Ind valid and test sets, OoD set
    """
    n_samples_bdd_valid = int(bdd_valid_mc_samples.shape[0] / precomputed_mcd_runs)
    n_samples_bdd_test = int(bdd_test_mc_samples.shape[0] / precomputed_mcd_runs)
    n_samples_ood = int(ood_test_mc_samples.shape[0] / precomputed_mcd_runs)
    # Reshape to easily subset mcd samples
    reshaped_bdd_valid = bdd_valid_mc_samples.reshape(n_samples_bdd_valid,
                                                      precomputed_mcd_runs,
                                                      bdd_valid_mc_samples.shape[1])
    reshaped_bdd_test = bdd_test_mc_samples.reshape(n_samples_bdd_test,
                                                    precomputed_mcd_runs,
                                                    bdd_test_mc_samples.shape[1])
    reshaped_ood_test = ood_test_mc_samples.reshape(n_samples_ood,
                                                    precomputed_mcd_runs,
                                                    ood_test_mc_samples.shape[1])
    # Select the desired number of samples to take
    bdd_valid_mc_samples = reshaped_bdd_valid[:, :n_mcd_runs, :].reshape(n_samples_bdd_valid * n_mcd_runs,
                                                                         bdd_valid_mc_samples.shape[1])
    bdd_test_mc_samples = reshaped_bdd_test[:, :n_mcd_runs, :].reshape(n_samples_bdd_test * n_mcd_runs,
                                                                       bdd_test_mc_samples.shape[1])
    ood_test_mc_samples = reshaped_ood_test[:, :n_mcd_runs, :].reshape(n_samples_ood * n_mcd_runs,
                                                                       ood_test_mc_samples.shape[1])
    return bdd_valid_mc_samples, bdd_test_mc_samples, ood_test_mc_samples


def get_input_transformations(cifar10_normalize_inputs: bool, img_size: int, extra_augmentations: bool):
    if cifar10_normalize_inputs:
        if extra_augmentations:
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.)
                        ]),
                        p=0.3
                    ),
                    torchvision.transforms.RandomGrayscale(p=0.1),
                    torchvision.transforms.RandomVerticalFlip(p=0.4),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.01, 0.2))
                        ]),
                        p=0.3
                    ),
                    cifar10_normalization(),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    cifar10_normalization(),
                    torchvision.transforms.ToTensor(),
                ]
            )
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(img_size, img_size)),
                cifar10_normalization(),
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        if extra_augmentations:
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.)
                        ]),
                        p=0.3
                    ),
                    torchvision.transforms.RandomGrayscale(p=0.1),
                    torchvision.transforms.RandomVerticalFlip(p=0.4),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.01, 0.2))
                        ]),
                        p=0.3
                    ),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            )

        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(img_size, img_size)),
                torchvision.transforms.ToTensor(),
            ]
        )

    return train_transforms, test_transforms
