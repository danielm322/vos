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
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
from detectron2.engine import launch
# from detectron2.data import build_detection_test_loader, MetadataCatalog
# Project imports
# from core.evaluation_tools.evaluation_utils import get_train_contiguous_id_to_test_thing_dataset_id_dict
from core.setup import setup_config, setup_arg_parser
from inference.inference_utils import instances_to_json, get_inference_output_dir, build_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from detectron2.data.detection_utils import read_image
# Latent space OOD detection imports
# The following matplotlib backend (TkAgg) seems to be the only one that easily can plot either on the main or in
# the second screen. Remove or change the matplotlib backend if it doesn't work well
import matplotlib
matplotlib.use('TkAgg')
from ls_ood_detect_cea.uncertainty_estimation import Hook
from ls_ood_detect_cea.uncertainty_estimation import deeplabv3p_apply_dropout
from ls_ood_detect_cea.uncertainty_estimation import get_dl_h_z
from ls_ood_detect_cea import plot_samples_pacmap, apply_pca_ds_split, apply_pca_transform, DetectorKDE, get_hz_scores, \
    get_hz_detector_results, plot_roc_ood_detector
from detection.TDL_helper_functions import build_in_distribution_valid_test_dataloader_args, build_data_loader, \
    build_ood_dataloader_args, get_ls_mcd_samples, fit_pacmap, apply_pacmap_transform


def main(args):
    # Setup config
    cfg = setup_config(args,
                       random_seed=args.random_seed,
                       is_testing=True)
    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    # cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type
    # Check if the inference is using vos, by checking if the config file is vos.yaml
    # Returns False otherwise
    using_vos = args.config_file.split('.yaml')[0].split('/')[-1] == 'vos'
    # Set up number of cpu threads#
    # torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Create inference output directory and copy inference config file to keep
    # track of experimental settings
    inference_output_dir = get_inference_output_dir(
        cfg['OUTPUT_DIR'],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level)

    os.makedirs(inference_output_dir, exist_ok=True)
    copyfile(args.inference_config, os.path.join(
        inference_output_dir, os.path.split(args.inference_config)[-1]))

    # Get category mapping dictionary:
    # train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
    #     cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
    # test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
    #     args.test_dataset).thing_dataset_id_to_contiguous_id

    # If both dicts are equal or if we are performing out of distribution
    # detection, just flip the test dict.
    # cat_mapping_dict = get_train_contiguous_id_to_test_thing_dataset_id_dict(
    #     cfg,
    #     args,
    #     train_thing_dataset_id_to_contiguous_id,
    #     test_thing_dataset_id_to_contiguous_id)

    # Build predictor
    predictor = build_predictor(cfg)
    # Place the Hook at the output of the last dropout layer
    hooked_dropout_layer = Hook(predictor.model.roi_heads.box_head)
    # Put model in evaluation mode
    predictor.model.eval()
    # Activate Dropout layers
    predictor.model.apply(deeplabv3p_apply_dropout)
    # Build In Distribution valid adn test data loader
    ind_valid_dl_args, ind_test_dl_args = build_in_distribution_valid_test_dataloader_args(
        cfg,
        dataset_name=args.test_dataset,
        split_proportion=0.8
    )
    ind_valid_dl = build_data_loader(**ind_valid_dl_args)
    ind_test_dl = build_data_loader(**ind_test_dl_args)
    del ind_valid_dl_args
    del ind_test_dl_args
    # ind_data_loader = build_detection_test_loader(
    #     cfg, dataset_name=args.test_dataset)

    # Build Out of Distribution test data loader
    ood_data_loader_args = build_ood_dataloader_args(cfg)
    ood_test_data_loader = build_data_loader(**ood_data_loader_args)
    del ood_data_loader_args

    # Get Monte-Carlo samples
    bdd_valid_mc_samples = get_ls_mcd_samples(model=predictor,
                                              data_loader=ind_valid_dl,
                                              mcd_nro_samples=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
                                              hook_dropout_layer=hooked_dropout_layer,
                                              layer_type=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE)
    del ind_valid_dl
    # Save MC samples if specified
    if cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.SAVE_MC_SAMPLES:
        num_images_to_save = int(bdd_valid_mc_samples.shape[0] / cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS)
        torch.save(bdd_valid_mc_samples,
                   f"./bdd_valid_{num_images_to_save}_{bdd_valid_mc_samples.shape[1]}_mcd_samples.pt")
    # Get Monte-Carlo samples
    bdd_test_mc_samples = get_ls_mcd_samples(model=predictor,
                                             data_loader=ind_test_dl,
                                             mcd_nro_samples=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
                                             hook_dropout_layer=hooked_dropout_layer,
                                             layer_type=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE)
    del ind_test_dl
    # Save MC samples if specified
    if cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.SAVE_MC_SAMPLES:
        num_images_to_save = int(bdd_test_mc_samples.shape[0] / cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS)
        torch.save(bdd_test_mc_samples,
                   f"./bdd_test_{num_images_to_save}_{bdd_test_mc_samples.shape[1]}_mcd_samples.pt")
    # Get Monte-Carlo samples
    ood_test_mc_samples = get_ls_mcd_samples(model=predictor,
                                             data_loader=ood_test_data_loader,
                                             mcd_nro_samples=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
                                             hook_dropout_layer=hooked_dropout_layer,
                                             layer_type=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE)
    del ood_test_data_loader
    # Save MC samples if specified
    if cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.SAVE_MC_SAMPLES:
        num_images_to_save = int(ood_test_mc_samples.shape[0] / cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS)
        torch.save(ood_test_mc_samples,
                   f"./ood_test_{num_images_to_save}_{ood_test_mc_samples.shape[1]}_mcd_samples.pt")

    # Calculate entropy for bdd valid set
    _, bdd_valid_h_z_np = get_dl_h_z(bdd_valid_mc_samples,
                                     mcd_samples_nro=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS)
    # Save entropy calculations if specified
    if cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.SAVE_ENTROPY_SAMPLES:
        np.save(f"./bdd_valid_{bdd_valid_h_z_np.shape[0]}_{bdd_valid_h_z_np.shape[1]}_h_z_samples",
                bdd_valid_h_z_np)
    # Calculate entropy bdd test set
    _, bdd_test_h_z_np = get_dl_h_z(bdd_test_mc_samples,
                                    mcd_samples_nro=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS)
    # Save entropy calculations if specified
    if cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.SAVE_ENTROPY_SAMPLES:
        np.save(f"./bdd_test_{bdd_test_h_z_np.shape[0]}_{bdd_test_h_z_np.shape[1]}_h_z_samples",
                bdd_test_h_z_np)
    # Calculate entropy ood test set
    _, ood_h_z_np = get_dl_h_z(ood_test_mc_samples,
                               mcd_samples_nro=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS)
    # Save entropy calculations if specified
    if cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.SAVE_ENTROPY_SAMPLES:
        np.save(f"./ood_test_{ood_h_z_np.shape[0]}_{ood_h_z_np.shape[1]}_h_z_samples",
                ood_h_z_np)
    # Slurm machine cannot plot, so we should avoid plotting on this machine
    if not args.slurm_inference:
        # Check entropy 2D projection
        plot_samples_pacmap(samples_ind=bdd_test_h_z_np,
                            samples_ood=ood_h_z_np,
                            neighbors=28,
                            title="BDD - COCO: $\hat{H}_{\phi}(z_i \mid x)$")

    # Perform PCA dimension reduction
    pca_h_z_bdd_valid_samples, pca_transformation = apply_pca_ds_split(samples=bdd_valid_h_z_np,
                                                                       nro_components=16)
    pca_h_z_bdd_test_samples = apply_pca_transform(bdd_test_h_z_np, pca_transformation)
    pca_h_z_ood_samples = apply_pca_transform(ood_h_z_np, pca_transformation)

    if not args.slurm_inference:
        # Check PCA separation with pacmap
        plot_samples_pacmap(samples_ind=pca_h_z_bdd_test_samples,
                            samples_ood=pca_h_z_ood_samples,
                            neighbors=16,
                            title="$\hat{H}_{\phi}(z_i \mid x)$ PCA test: BDD vs COCO")
        plot_samples_pacmap(samples_ind=pca_h_z_bdd_valid_samples,
                            samples_ood=pca_h_z_ood_samples,
                            neighbors=16,
                            title="$\hat{H}_{\phi}(z_i \mid x)$ PCA valid: BDD vs COCO")

    # Apply pacmap transform
    pm_h_z_bdd_valid_samples, pm_transformation = fit_pacmap(samples_ind=bdd_valid_h_z_np,
                                                             neighbors=25,
                                                             components=2)
    pm_h_z_bdd_test_samples = apply_pacmap_transform(new_samples=bdd_test_h_z_np,
                                                     original_samples=bdd_valid_h_z_np,
                                                     pm_instance=pm_transformation)
    pm_h_z_ood_samples = apply_pacmap_transform(new_samples=ood_h_z_np,
                                                original_samples=bdd_valid_h_z_np,
                                                pm_instance=pm_transformation)

    # Build OoD detector
    bdd_ds_shift_detector = DetectorKDE(train_embeddings=pm_h_z_bdd_valid_samples)
    scores_bdd_test = bdd_ds_shift_detector.density.score_samples(pm_h_z_bdd_test_samples)
    scores_ood = bdd_ds_shift_detector.density.score_samples(pm_h_z_ood_samples)
    # scores_bdd_test = get_hz_scores(hz_detector=bdd_ds_shift_detector,
    #                                 samples=pm_h_z_bdd_test_samples)
    # scores_ood = get_hz_scores(hz_detector=bdd_ds_shift_detector,
    #                            samples=pm_h_z_ood_samples)
    results_ood = get_hz_detector_results(detect_exp_name="BDD test vs. COCO ood",
                                          ind_samples_scores=scores_bdd_test,
                                          ood_samples_scores=scores_ood)
    if not args.slurm_inference:
        plot_roc_ood_detector(results_table=results_ood,
                              legend_title="COCO OoD detection",
                              plot_title="ROC Curve COCO OoD Detection")


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
