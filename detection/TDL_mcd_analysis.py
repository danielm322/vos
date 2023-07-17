import mlflow
import numpy as np
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from os.path import join as op_join
from pytorch_lightning import seed_everything as pl_seed_everything
from ls_ood_detect_cea.uncertainty_estimation import get_dl_h_z
from ls_ood_detect_cea import plot_samples_pacmap, apply_pca_ds_split, apply_pca_transform, \
    get_hz_detector_results, save_roc_ood_detector, fit_pacmap, apply_pacmap_transform
from tqdm import tqdm
from TDL_param_logging import log_params_from_omegaconf_dict
from TDL_mcd_helper_fns import fit_evaluate_KDE, adjust_mlflow_results_name, reduce_mcd_samples
import warnings
# Filter the append warning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

# If both next two flags are false, mlflow will create a local tracking uri for the experiment
# Upload analysis to the TDL server
UPLOAD_FROM_LOCAL_TO_SERVER = True
# Upload analysis ran on the TDL server
UPLOAD_FROM_SERVER_TO_SERVER = False
assert UPLOAD_FROM_SERVER_TO_SERVER + UPLOAD_FROM_LOCAL_TO_SERVER <= 1
# Perform analysis either on RCNN or RESNET
RCNN = True
RESNET = False
assert RCNN + RESNET == 1
if RCNN:
    config_file = "config_rcnn.yaml"
else:
    config_file = "config.yaml"


@hydra.main(version_base=None, config_path="configs/MCD_evaluation", config_name=config_file)
def main(cfg: DictConfig) -> None:
    """
    This function performs analysis on already calculated MCD samples in another script.
    Evaluates BDD as In distribution dataset against either COCO or Openimages as described in the VOS repository.
    This script assumes without checking that the number of MCD runs is exactly the same for the
    InD (BDD) data and the OoD data.
    :return: None
    """
    ############################
    #      Seed Everything     #
    ############################
    pl_seed_everything(cfg.seed)
    # Select device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ###############################################
    # Load precalculated MCD samples              #
    ###############################################
    # Inspect correct naming of files and dataset
    assert cfg.ood_dataset in cfg.ood_mcd_samples and cfg.ood_dataset in cfg.ood_entropy_test, "OoO Dataset name and preloaded files must coincide"
    assert cfg.layer_type in cfg.ood_mcd_samples and cfg.layer_type in cfg.ood_entropy_test, "Location of samples must coincide with filename"
    assert "h_z" in cfg.ood_entropy_test and "h_z" not in cfg.ood_mcd_samples
    assert cfg.ood_dataset in ('coco', 'openimages', "gtsrb", "svhn", "cifar10")
    bdd_valid_mc_samples = torch.load(f=op_join(cfg.data_dir, cfg.bdd_valid_mcd_samples),
                                      map_location=device)
    bdd_test_mc_samples = torch.load(f=op_join(cfg.data_dir, cfg.bdd_test_mcd_samples),
                                     map_location=device)
    ood_test_mc_samples = torch.load(f=op_join(cfg.data_dir, cfg.ood_mcd_samples),
                                     map_location=device)

    ##############################################################
    # Select number of MCD runs to use                           #
    ##############################################################
    assert cfg.n_mcd_runs <= cfg.precomputed_mcd_runs, "n_mcd_runs must be less than or equal to the precomputed runs"
    if cfg.n_mcd_runs < cfg.precomputed_mcd_runs:
        bdd_valid_mc_samples, bdd_test_mc_samples, ood_test_mc_samples = reduce_mcd_samples(
            bdd_valid_mc_samples=bdd_valid_mc_samples,
            bdd_test_mc_samples=bdd_test_mc_samples,
            ood_test_mc_samples=ood_test_mc_samples,
            precomputed_mcd_runs=cfg.precomputed_mcd_runs,
            n_mcd_runs=cfg.n_mcd_runs
        )

    #################################################################
    # Select number of proposals from RPN to use
    #################################################################
    max_n_proposals = cfg.max_n_proposals
    assert cfg.use_n_proposals <= max_n_proposals, "use_n_proposals must be less than or equal to the max proposals"
    # Compute entropy only of n_mcd_runs < cfg.precomputed_mcd_runs, otherwise, load precomputed entropy
    if cfg.use_n_proposals < max_n_proposals and cfg.n_mcd_runs < cfg.precomputed_mcd_runs:
        # Randomly select the columns to keep:
        columns_to_keep = torch.randperm(max_n_proposals)[:cfg.use_n_proposals]
        bdd_valid_mc_samples = bdd_valid_mc_samples[:, columns_to_keep]
        bdd_test_mc_samples = bdd_test_mc_samples[:, columns_to_keep]
        ood_test_mc_samples = ood_test_mc_samples[:, columns_to_keep]

    #####################################################################
    # Compute entropy
    #####################################################################
    if cfg.n_mcd_runs == cfg.precomputed_mcd_runs:
        # Load precomputed entropy in this case
        bdd_valid_h_z_np = np.load(file=op_join(cfg.data_dir, cfg.bdd_entropy_valid))
        bdd_test_h_z_np = np.load(file=op_join(cfg.data_dir, cfg.bdd_entropy_test))
        ood_h_z_np = np.load(file=op_join(cfg.data_dir, cfg.ood_entropy_test))
        # Reduce number of proposals here, to avoid recalculations
        if cfg.use_n_proposals < max_n_proposals:
            columns_to_keep = torch.randperm(max_n_proposals)[:cfg.use_n_proposals]
            bdd_valid_h_z_np = bdd_valid_h_z_np[:, columns_to_keep]
            bdd_test_h_z_np = bdd_test_h_z_np[:, columns_to_keep]
            ood_h_z_np = ood_h_z_np[:, columns_to_keep]
    # Calculate entropy only if cfg.n_mcd_runs < cfg.precomputed_mcd_runs
    else:
        # Calculate entropy for bdd valid set
        _, bdd_valid_h_z_np = get_dl_h_z(bdd_valid_mc_samples,
                                         mcd_samples_nro=cfg.n_mcd_runs)
        # Calculate entropy bdd test set
        _, bdd_test_h_z_np = get_dl_h_z(bdd_test_mc_samples,
                                        mcd_samples_nro=cfg.n_mcd_runs)
        # Calculate entropy ood test set
        _, ood_h_z_np = get_dl_h_z(ood_test_mc_samples,
                                   mcd_samples_nro=cfg.n_mcd_runs)

    # Since these data is no longer needed we can delete it
    del bdd_valid_mc_samples
    del bdd_test_mc_samples
    del ood_test_mc_samples

    #######################################################################
    # Setup MLFLow
    #######################################################################
    # Setup MLFlow for experiment tracking
    # MlFlow configuration
    experiment_name = cfg.logger.mlflow.experiment_name
    if UPLOAD_FROM_LOCAL_TO_SERVER:
        mlflow.set_tracking_uri("http://10.8.33.50:5050")
    elif UPLOAD_FROM_SERVER_TO_SERVER:
        mlflow.set_tracking_uri("http://127.0.0.1:5051")
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(
            name=experiment_name,
        )
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    # mlflow.set_tracking_uri(cfg.logger.mlflow.tracking_uri)
    # Let us define a run name automatically.
    if cfg.ood_dataset == "coco":
        mlflow_run_dataset = "cc"
    elif cfg.ood_dataset == "openimages":
        mlflow_run_dataset = "oi"
    elif cfg.ood_dataset == "gtsrb":
        mlflow_run_dataset = "gtsrb"
    elif cfg.ood_dataset == "svhn":
        mlflow_run_dataset = "svhn"
    else:
        mlflow_run_dataset = "cifar10"
    mlflow_run_name = f"{mlflow_run_dataset}_{cfg.layer_type}_{cfg.n_mcd_runs}_mcd_{cfg.use_n_proposals}"

    ##########################################################################
    # Start the evaluation run
    ##########################################################################
    # Define mlflow run to log metrics and parameters
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=mlflow_run_name) as run:
        # Log parameters with mlflow
        log_params_from_omegaconf_dict(cfg)
        # Check entropy 2D projection
        pacmap_2d_proj_plot = plot_samples_pacmap(samples_ind=bdd_test_h_z_np,
                                                  samples_ood=ood_h_z_np,
                                                  neighbors=cfg.n_pacmap_neighbors,
                                                  title=cfg.ind_dataset + " - " + cfg.ood_dataset + " : $\hat{H}_{\phi}(z_i \mid x)$",
                                                  return_figure=True)
        mlflow.log_figure(figure=pacmap_2d_proj_plot,
                          artifact_file="figs/h_z_pacmap.png")
        #######################################################################
        # Start PCA evaluation
        #######################################################################
        pca_metrics = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                            'fpr', 'tpr', 'roc_thresholds',
                                            'precision', 'recall', 'pr_thresholds'])
        for n_components in tqdm(cfg.n_pca_components, desc="Evaluating PCA"):
            # Perform PCA dimension reduction
            pca_h_z_bdd_valid_samples, pca_transformation = apply_pca_ds_split(samples=bdd_valid_h_z_np,
                                                                               nro_components=n_components)
            pca_h_z_bdd_test_samples = apply_pca_transform(bdd_test_h_z_np, pca_transformation)
            pca_h_z_ood_samples = apply_pca_transform(ood_h_z_np, pca_transformation)

            # Build OoD detector
            scores_bdd_test, scores_ood = fit_evaluate_KDE(h_z_ind_valid_samples=pca_h_z_bdd_valid_samples,
                                                           h_z_ind_test_samples=pca_h_z_bdd_test_samples,
                                                           h_z_ood_samples=pca_h_z_ood_samples,
                                                           normalize=cfg.normalize_kde_prediction,
                                                           transform_exp=cfg.exp_kde)

            results_ood, results_for_mlflow = get_hz_detector_results(detect_exp_name=f"PCA_{n_components}_components",
                                                                      ind_samples_scores=scores_bdd_test,
                                                                      ood_samples_scores=scores_ood,
                                                                      return_results_for_mlflow=True)
            results_for_mlflow = adjust_mlflow_results_name(data_dict=results_for_mlflow, technique_name='pca')

            mlflow.log_metrics(results_for_mlflow, step=n_components)
            pca_metrics = pca_metrics.append(results_ood)
        # Plot all PCA evaluations in one figure
        roc_curves_pca = save_roc_ood_detector(
            results_table=pca_metrics,
            plot_title=f"ROC {cfg.ind_dataset} vs {cfg.ood_dataset} OoD Detection PCA {cfg.layer_type}"
        )
        # Log the plot with mlflow
        mlflow.log_figure(figure=roc_curves_pca,
                          artifact_file="figs/roc_curves_pca.png")

        ##########################################################################
        # Start PacMAP evaluation
        ##########################################################################
        pacmap_metrics = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                               'fpr', 'tpr', 'roc_thresholds',
                                               'precision', 'recall', 'pr_thresholds'])
        # Evaluate number of neighbors to build the algorithm
        for n_components in tqdm(cfg.n_pacmap_components, desc="Evaluating PacMAP"):
            # Apply pacmap transform
            pm_h_z_bdd_valid_samples, pm_transformation = fit_pacmap(samples_ind=bdd_valid_h_z_np,
                                                                     neighbors=cfg.n_pacmap_neighbors,
                                                                     components=n_components)
            pm_h_z_bdd_test_samples = apply_pacmap_transform(new_samples=bdd_test_h_z_np,
                                                             original_samples=bdd_valid_h_z_np,
                                                             pm_instance=pm_transformation)
            pm_h_z_ood_samples = apply_pacmap_transform(new_samples=ood_h_z_np,
                                                        original_samples=bdd_valid_h_z_np,
                                                        pm_instance=pm_transformation)

            # Build and evaluate OoD detector
            scores_bdd_test, scores_ood = fit_evaluate_KDE(h_z_ind_valid_samples=pm_h_z_bdd_valid_samples,
                                                           h_z_ind_test_samples=pm_h_z_bdd_test_samples,
                                                           h_z_ood_samples=pm_h_z_ood_samples,
                                                           normalize=cfg.normalize_kde_prediction,
                                                           transform_exp=cfg.exp_kde)
            results_ood, results_for_mlflow = get_hz_detector_results(detect_exp_name=f"PacMAP_{n_components}_components",
                                                                      ind_samples_scores=scores_bdd_test,
                                                                      ood_samples_scores=scores_ood,
                                                                      return_results_for_mlflow=True)
            results_for_mlflow = adjust_mlflow_results_name(data_dict=results_for_mlflow, technique_name='pm')

            mlflow.log_metrics(results_for_mlflow, step=n_components)
            pacmap_metrics = pacmap_metrics.append(results_ood)

        # Plot all PacMAP evaluations in one figure
        roc_curves_pacmap = save_roc_ood_detector(
            results_table=pacmap_metrics,
            plot_title=f"ROC {cfg.ind_dataset} vs {cfg.ood_dataset} OoD Detection PacMAP {cfg.layer_type}"
        )
        # Log the plot with mlflow
        mlflow.log_figure(figure=roc_curves_pacmap,
                          artifact_file="figs/roc_curves_pacmap.png")

        mlflow.end_run()


if __name__ == "__main__":
    main()
