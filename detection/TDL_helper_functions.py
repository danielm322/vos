from typing import Union, List, Any, Callable, Dict, Optional, Tuple
import numpy as np
import pacmap
import torch
import torch.utils.data as torchdata
from tqdm import tqdm
from ls_ood_detect_cea.uncertainty_estimation import Hook
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import DatasetMapper, DatasetFromList, MapDataset
from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers import InferenceSampler


# Get latent space Monte Carlo Dropout samples
def get_ls_mcd_samples(model: torch.nn.Module,
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
    assert layer_type in ("FC", "Conv"), "Layer type must be either 'FC' or 'Conv'"
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
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
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=1)
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
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
                if i == 29:
                    break
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


def fit_pacmap(samples_ind: np.array,
               neighbors: int = 25,
               components: int = 2
               ) -> Tuple[np.array, pacmap.PaCMAP]:
    """
    In-Distribution vs Out-of-Distribution Data Projection 2D Plot using PaCMAP algorithm.

    :param components: Number of components in the output
    :type components: int
    :param samples_ind: In-Distribution (InD) samples numpy array
    :type samples_ind: np.ndarray
    :param neighbors: Number of nearest-neighbors considered for the PaCMAP algorithm
    :type neighbors: int
    :return:
    :rtype: None
    """
    embedding = pacmap.PaCMAP(n_components=components, n_neighbors=neighbors, MN_ratio=0.5, FP_ratio=2.0)
    samples_transformed = embedding.fit_transform(samples_ind, init="pca")
    return samples_transformed, embedding


def apply_pacmap_transform(new_samples: np.array,
                           original_samples: np.array,
                           pm_instance: pacmap.PaCMAP) -> np.array:
    return pm_instance.transform(X=new_samples, basis=original_samples)
