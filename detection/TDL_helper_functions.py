from typing import Union, List, Any, Callable, Dict, Optional, Tuple
import numpy as np
import torch
import torch.utils.data as torchdata
from dropblock import DropBlock2D
from ls_ood_detect_cea import DetectorKDE, get_hz_scores
from tqdm import tqdm
from ls_ood_detect_cea.uncertainty_estimation import Hook
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import DatasetMapper, DatasetFromList, MapDataset
from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers import InferenceSampler

# Ugly but fast way to test to hook the backbone: get raw output, apply dropblock
# inside the following function
dropblock_ext = DropBlock2D(drop_prob=0.5, block_size=3)


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
                        latent_mcd_sample = latent_mcd_sample.flatten()
                    elif layer_type == "backbone":
                        # Apply dropblock
                        for k, v in latent_mcd_sample.items():
                            latent_mcd_sample[k] = dropblock_ext(v)
                            # Get image HxW mean:
                            latent_mcd_sample[k] = torch.mean(latent_mcd_sample[k], dim=2, keepdim=True)
                            latent_mcd_sample[k] = torch.mean(latent_mcd_sample[k], dim=3, keepdim=True)
                            # Remove useless dimensions:
                            latent_mcd_sample[k] = torch.squeeze(latent_mcd_sample[k], dim=3)
                            latent_mcd_sample[k] = torch.squeeze(latent_mcd_sample[k], dim=2)
                        latent_mcd_sample = torch.cat(list(latent_mcd_sample.values()), dim=1)
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


def build_data_loader(
        dataset: Union[List[Any], torchdata.Dataset],
        mapper: Callable[[Dict[str, Any]], Any],
        sampler: Optional[torchdata.Sampler] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def build_ood_dataloader_args(cfg):
    """
    Builds the OOD dataset from a cfg file argument in cfg.PROBABILISTIC_INFERENCE.OOD_DATASET for OOD set.
    Assumes the dataset will be correctly found with this only argument
    :param cfg: Configuration class parameters
    :return: Dictionary of dataset, mapper, num_workers, sampler
    """
    dataset = get_detection_dataset_dicts(names=cfg.PROBABILISTIC_INFERENCE.OOD_DATASET,
                                          filter_empty=False,
                                          proposal_files=None,
                                          )
    mapper = DatasetMapper(cfg, False)

    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(dataset))
        if not isinstance(dataset, torchdata.IterableDataset)
        else None,
    }


def build_in_distribution_valid_test_dataloader_args(cfg,
                                                     dataset_name: str,
                                                     split_proportion: float) -> Tuple[Dict, Dict]:
    """
    Builds the arguments (datasets, mappers, samplers) for the validation and test sets starting form a single
    validation split set.
    :param cfg: Configuration class parameters
    :param dataset_name:
    :param split_proportion: Sets the proportion of the validation set
    :return: Tuple of two dictionaries to build the validation and test set dataloaders
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    # Split dataset
    indexes_np = np.arange(len(dataset))
    np.random.shuffle(indexes_np)
    max_idx_valid = int(split_proportion * len(dataset))
    valid_idxs, test_idxs = indexes_np[:max_idx_valid], indexes_np[max_idx_valid:]
    valid_dataset, test_dataset = [dataset[i] for i in valid_idxs], [dataset[i] for i in test_idxs]

    mapper_val = DatasetMapper(cfg, False)
    mapper_test = DatasetMapper(cfg, False)
    return {
        "dataset": valid_dataset,
        "mapper": mapper_val,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(valid_dataset))
        if not isinstance(valid_dataset, torchdata.IterableDataset)
        else None,
    }, {"dataset": test_dataset,
        "mapper": mapper_test,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(test_dataset))
        if not isinstance(test_dataset, torchdata.IterableDataset)
        else None, }


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
