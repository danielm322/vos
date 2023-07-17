"""
Script for performing Probabilistic inference using MC Dropout and testing the ood detection
"""
import numpy as np

import core
import os
import sys
import torch
from shutil import copyfile

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), "src", "detr"))

# Detectron imports
from detectron2.engine import launch

# Project imports
# from core.evaluation_tools.evaluation_utils import get_train_contiguous_id_to_test_thing_dataset_id_dict
from core.setup import setup_config, setup_arg_parser
from inference.inference_utils import get_inference_output_dir, build_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from detectron2.data.detection_utils import read_image
# Latent space OOD detection imports
# The following matplotlib backend (TkAgg) seems to be the only one that easily can plot either on the main or in
# the second screen. Remove or change the matplotlib backend if it doesn't work well
# import matplotlib
# matplotlib.use('TkAgg')
from ls_ood_detect_cea.uncertainty_estimation import Hook
from ls_ood_detect_cea.uncertainty_estimation import deeplabv3p_apply_dropout
from ls_ood_detect_cea.uncertainty_estimation import get_dl_h_z
from TDL_helper_functions import (
    build_in_distribution_valid_test_dataloader_args,
    build_data_loader,
    build_ood_dataloader_args,
)
from TDL_mcd_helper_fns import get_ls_mcd_samples_rcnn


def main(args) -> None:
    """
    The current script has as only purpose to get the Monte Carlo Dropout samples, save them,
    and then calculate the entropy and save those quantities for further analysis. This will do this for the InD BDD set
    and one chosen OoD set
    :param args: Configuration class parameters
    :return: None
    """
    # Setup config
    cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)
    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    # cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type
    # Set up number of cpu threads#
    # torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Create inference output directory and copy inference config file to keep
    # track of experimental settings
    inference_output_dir = get_inference_output_dir(
        cfg["OUTPUT_DIR"],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level,
    )

    os.makedirs(inference_output_dir, exist_ok=True)
    copyfile(
        args.inference_config,
        os.path.join(inference_output_dir, os.path.split(args.inference_config)[-1]),
    )
    # Samples save folder
    SAVE_FOLDER = f"./MCD_evaluation_data/{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}/"
    # Assert only one layer is specified to be hooked
    assert (
        cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_RELU_AFTER_DROPOUT
        + cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPOUT_BEFORE_RELU
        + cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPBLOCK_RPN
        + cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPBLOCK_AFTER_BACKBONE
        == 1
    ), " Select only one layer to be hooked"
    ##################################################################################
    # Prepare predictor and data loaders
    ##################################################################################
    # Build predictor
    predictor = build_predictor(cfg)
    if cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_RELU_AFTER_DROPOUT:
        # Hook the final activation of the module: the ReLU after the dropout
        hooked_dropout_layer = Hook(predictor.model.roi_heads.box_head)
    elif cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPOUT_BEFORE_RELU:
        # Place the Hook at the output of the last dropout layer
        hooked_dropout_layer = Hook(predictor.model.roi_heads.box_head.fc_dropout2)
    elif cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPBLOCK_RPN:
        hooked_dropout_layer = Hook(
            predictor.model.proposal_generator.rpn_head.dropblock
        )
    elif cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPBLOCK_AFTER_BACKBONE:
        hooked_dropout_layer = Hook(
            predictor.model.backbone
        )
    # Put model in evaluation mode
    predictor.model.eval()
    # Activate Dropout layers
    predictor.model.apply(deeplabv3p_apply_dropout)
    # Build In Distribution valid and test data loader
    (
        ind_valid_dl_args,
        ind_test_dl_args,
    ) = build_in_distribution_valid_test_dataloader_args(
        cfg, dataset_name=args.test_dataset, split_proportion=0.8
    )
    ind_valid_dl = build_data_loader(**ind_valid_dl_args)
    ind_test_dl = build_data_loader(**ind_test_dl_args)
    del ind_valid_dl_args
    del ind_test_dl_args

    # Build Out of Distribution test data loader
    ood_data_loader_args = build_ood_dataloader_args(cfg)
    ood_test_data_loader = build_data_loader(**ood_data_loader_args)
    del ood_data_loader_args

    ###################################################################################################
    # Perform MCD inference and save samples
    ###################################################################################################
    # Get Monte-Carlo samples
    bdd_valid_mc_samples = get_ls_mcd_samples_rcnn(
        model=predictor,
        data_loader=ind_valid_dl,
        mcd_nro_samples=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
        hook_dropout_layer=hooked_dropout_layer,
        layer_type=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE,
    )
    del ind_valid_dl
    # Save MC samples
    num_images_to_save = int(
        bdd_valid_mc_samples.shape[0] / cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS
    )
    torch.save(
        bdd_valid_mc_samples,
        f"./{SAVE_FOLDER}/bdd_valid_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}_{num_images_to_save}_{bdd_valid_mc_samples.shape[1]}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS}_mcd_samples.pt",
    )
    # Get Monte-Carlo samples
    bdd_test_mc_samples = get_ls_mcd_samples_rcnn(
        model=predictor,
        data_loader=ind_test_dl,
        mcd_nro_samples=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
        hook_dropout_layer=hooked_dropout_layer,
        layer_type=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE,
    )
    del ind_test_dl
    # Save MC samples
    num_images_to_save = int(
        bdd_test_mc_samples.shape[0] / cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS
    )
    torch.save(
        bdd_test_mc_samples,
        f"./{SAVE_FOLDER}/bdd_test_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}_{num_images_to_save}_{bdd_test_mc_samples.shape[1]}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS}_mcd_samples.pt",
    )
    # Get Monte-Carlo samples
    ood_test_mc_samples = get_ls_mcd_samples_rcnn(
        model=predictor,
        data_loader=ood_test_data_loader,
        mcd_nro_samples=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
        hook_dropout_layer=hooked_dropout_layer,
        layer_type=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE,
    )
    del ood_test_data_loader
    # Save MC samples
    num_images_to_save = int(
        ood_test_mc_samples.shape[0] / cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS
    )
    torch.save(
        ood_test_mc_samples,
        f"./{SAVE_FOLDER}/ood_test_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}_{num_images_to_save}_{ood_test_mc_samples.shape[1]}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS}_mcd_samples.pt",
    )
    # Since inference if memory-intense, we want to liberate as much memory as possible
    del predictor
    ########################################################################################
    # Calculate and save entropy
    ########################################################################################
    # Calculate entropy for bdd valid set
    _, bdd_valid_h_z_np = get_dl_h_z(
        bdd_valid_mc_samples,
        mcd_samples_nro=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
    )
    # Save entropy calculations
    np.save(
        f"./{SAVE_FOLDER}/bdd_valid_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}_{bdd_valid_h_z_np.shape[0]}_{bdd_valid_h_z_np.shape[1]}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS}_mcd_h_z_samples",
        bdd_valid_h_z_np,
    )
    # Calculate entropy bdd test set
    _, bdd_test_h_z_np = get_dl_h_z(
        bdd_test_mc_samples,
        mcd_samples_nro=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
    )
    # Save entropy calculations
    np.save(
        f"./{SAVE_FOLDER}/bdd_test_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}_{bdd_test_h_z_np.shape[0]}_{bdd_test_h_z_np.shape[1]}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS}_mcd_h_z_samples",
        bdd_test_h_z_np,
    )
    # Calculate entropy ood test set
    _, ood_h_z_np = get_dl_h_z(
        ood_test_mc_samples,
        mcd_samples_nro=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
    )
    # Save entropy calculations
    np.save(
        f"./{SAVE_FOLDER}/ood_test_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}_{ood_h_z_np.shape[0]}_{ood_h_z_np.shape[1]}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS}_mcd_h_z_samples",
        ood_h_z_np,
    )
    # Analysis of the calculated samples is performed in another script!


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    print("Command Line Args:", args)
    # This function checks if there are multiple gpus, then it launches the distributed inference, otherwise it
    # just launches the main function, i.e., would act as a function wrapper passing the args to main
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
