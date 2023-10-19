import torch
import numpy as np
from ls_ood_detect_cea.rcnn import get_energy_score_rcnn
from torch.utils.data import DataLoader


def save_energy_scores_baselines(predictor: torch.nn.Module,
                                 ind_data_loader: DataLoader,
                                 ood_data_loader: DataLoader,
                                 baseline_name: str,
                                 save_foldar_name: str,
                                 ind_ds_name: str,
                                 ood_ds_name: str,
                                 ):
    print(f"\n{baseline_name} from InD {ind_ds_name}")
    ind_raw_test_energy, ind_filtered_test_energy = \
        get_energy_score_rcnn(dnn_model=predictor, input_dataloader=ind_data_loader)
    np.save(f"./{save_foldar_name}/{ind_ds_name}_ind_raw_{baseline_name}", ind_raw_test_energy)
    np.save(f"./{save_foldar_name}/{ind_ds_name}_ind_filtered_{baseline_name}", ind_filtered_test_energy)
    print(f"\n{baseline_name} from OoD {ood_ds_name}")
    # OoD
    ood_raw_test_energy, ood_filtered_test_energy = \
        get_energy_score_rcnn(dnn_model=predictor, input_dataloader=ood_data_loader)
    np.save(f"./{save_foldar_name}/{ood_ds_name}_ood_raw_{baseline_name}", ood_raw_test_energy)
    np.save(f"./{save_foldar_name}/{ood_ds_name}_ood_filtered_{baseline_name}", ood_filtered_test_energy)

