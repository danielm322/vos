#!/usr/bin/env python
# coding: utf-8
import mlflow
import hydra
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
import warnings
from ls_ood_detect_cea.uncertainty_estimation import Hook, deeplabv3p_apply_dropout
from TDL_smodel import FashionCNN
from detection.TDL_param_logging import log_params_from_omegaconf_dict
from detection.TDL_mcd_helper_fns import MCDSamplesExtractor, fit_evaluate_KDE, adjust_mlflow_results_name, \
    get_input_transformations
from ls_ood_detect_cea.uncertainty_estimation import get_dl_h_z
from ls_ood_detect_cea import (
    apply_pca_ds_split,
    apply_pca_transform,
    get_hz_detector_results,
    plot_samples_pacmap,
    save_roc_ood_detector, fit_pacmap, apply_pacmap_transform
)

# Filter the append warning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Upload analysis to the TDL server
UPLOAD_FROM_LOCAL_TO_SERVER = True
# Upload analysis ran on the TDL server
UPLOAD_FROM_SERVER_TO_SERVER = False
assert UPLOAD_FROM_SERVER_TO_SERVER + UPLOAD_FROM_LOCAL_TO_SERVER <= 1


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    assert cfg.hooked_layer in ("dropblock1", "dropblock2")
    # # Train model on Fashion MNIST
    train_transforms, test_transforms = get_input_transformations(
        cifar10_normalize_inputs=False,
        img_size=cfg.im_size,
        data_augmentations=cfg.data_augmentations
    )

    train_set = torchvision.datasets.FashionMNIST("./fashion_mnist_data", download=True, transform=train_transforms)
    init_test_set = torchvision.datasets.FashionMNIST("./fashion_mnist_data",
                                                      download=True,
                                                      train=False,
                                                      transform=test_transforms)
    torch.manual_seed(cfg.seed)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size)
    init_test_loader = DataLoader(init_test_set, batch_size=cfg.batch_size)

    # len(train_set), len(init_test_set)
    # Instantiate model
    model: FashionCNN = FashionCNN(dropblock_1=cfg.dropblock_1,
                                   dropblock_2=cfg.dropblock_2,
                                   dropout=cfg.dropout,
                                   dropblock_1_prob=cfg.dropblock_1_prob,
                                   dropblock_2_prob=cfg.dropblock_2_prob,
                                   dropblock_1_size=cfg.dropblock_1_size,
                                   dropblock_2_size=cfg.dropblock_2_size,
                                   dropout_prob=cfg.dropout_prob,
                                   leaky_relu=cfg.leaky_relu,
                                   spectral_normalization=cfg.spectral_normalization,
                                   average_pooling=cfg.average_pooling
                                   )
    model.to(device)
    # Loss fn
    error = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # print(model)

    #######################################################################
    # Setup MLFLow
    #######################################################################
    # Setup MLFlow for experiment tracking
    # MlFlow configuration
    experiment_name = cfg.experiment_name
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
    mlflow_run_dataset = "mnist"
    mlflow_run_name = f"{mlflow_run_dataset}_{cfg.layer_type}_{cfg.mcd_runs}_mcd"

    ##########################################################################
    # Start the evaluation run
    ##########################################################################
    # Define mlflow run to log metrics and parameters
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=mlflow_run_name) as run:
        # Log parameters with mlflow
        log_params_from_omegaconf_dict(cfg)

        ########################################################################
        # Train model
        ########################################################################
        count = 0
        # Lists for visualization of loss and accuracy
        loss_list = []
        iteration_list = []
        accuracy_list = []

        # Lists for knowing classwise accuracy
        predictions_list = []
        labels_list = []

        for epoch in range(cfg.num_epochs):
            for images, labels in train_loader:
                # Transfering images and labels to GPU if available
                images, labels = images.to(device), labels.to(device)

                train = Variable(images.view(cfg.batch_size, 1, cfg.im_size, cfg.im_size))
                labels = Variable(labels)

                # Forward pass
                outputs = model(train)
                loss = error(outputs, labels)

                # Initializing a gradient as 0 so there is no mixing of gradient among the batches
                optimizer.zero_grad()

                # Propagating the error backward
                loss.backward()

                # Optimizing the parameters
                optimizer.step()

                count += 1

                # Testing the model

                if not (count % 60):  # It's same as "if count % 50 == 0"
                    total = 0
                    correct = 0

                    for images, labels in init_test_loader:
                        images, labels = images.to(device), labels.to(device)
                        labels_list.append(labels)

                        test = Variable(images.view(cfg.batch_size, 1, cfg.im_size, cfg.im_size))

                        outputs = model(test)

                        predictions = torch.max(outputs, 1)[1].to(device)
                        predictions_list.append(predictions)
                        correct += (predictions == labels).sum()

                        total += len(labels)

                    accuracy = correct * 100 / total
                    mlflow.log_metric(key="accuracy", value=accuracy.cpu().item(), step=int(count // 60))
                    mlflow.log_metric(key="loss", value=loss.cpu().item(), step=int(count // 60))
                    loss_list.append(loss.data)
                    iteration_list.append(count)
                    accuracy_list.append(accuracy)

                if not (count % 600):
                    print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

        # Loss fig
        fig, axes = plt.subplots()
        axes.plot(iteration_list, [loss.cpu() for loss in loss_list], '.-')
        axes.set_xlabel("No. of Iteration")
        axes.set_ylabel("Loss")
        axes.set_title("Iterations vs Loss")
        mlflow.log_figure(figure=fig,
                          artifact_file="figs/loss.png")

        # Accuracy plot
        fig, axes = plt.subplots()
        axes.plot(iteration_list, [acc.cpu() for acc in accuracy_list], ".-")
        axes.set_xlabel("No. of Iteration")
        axes.set_ylabel("Accuracy")
        axes.set_title("Iterations vs Accuracy")
        mlflow.log_figure(figure=fig,
                          artifact_file="figs/accuracy.png")
        #####################################################################
        # # Get MCD Samples
        #####################################################################
        # With MNIST as OoD
        # Split test set into valid and test sets
        valid_set, test_set = train_test_split(init_test_set, test_size=0.2, random_state=42)
        valid_data_loader = DataLoader(valid_set, batch_size=1)
        test_data_loader = DataLoader(test_set, batch_size=1)

        # Load MNIST
        mnist_test_data = torchvision.datasets.MNIST('./mnist_data/', train=False, download=True,
                                                     transform=torchvision.transforms.Compose(
                                                         [torchvision.transforms.ToTensor()]))
        valid_set_mnist, test_set_mnist = train_test_split(mnist_test_data, test_size=0.2, random_state=42)
        # MNIST test set loader
        mnist_test_loader = DataLoader(test_set_mnist, batch_size=1, shuffle=True)
        # Hook layer
        if cfg.hooked_layer == "dropblock1":
            hooked_dropout_layer = Hook(model.dropblock1)
        else:
            hooked_dropout_layer = Hook(model.dropblock2)
        # Put model in evaluation mode
        model.eval()
        # Activate Dropout layers
        model.apply(deeplabv3p_apply_dropout)

        # Extract MCD samples
        mcd_extractor = MCDSamplesExtractor(
            model=model,
            mcd_nro_samples=cfg.mcd_runs,
            hook_dropout_layer=hooked_dropout_layer,
            layer_type=cfg.layer_type,
            device=device,
            architecture="small",
            location=2,
            reduction_method=cfg.reduction_method,
            input_size=cfg.im_size,
        )
        # InD Valid set
        fashion_valid_mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(valid_data_loader)
        # InD test set
        fashion_test_mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(test_data_loader)
        # ooD test set
        mnist_test_mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(mnist_test_loader)

        #########################################################################
        # Calculate entropy
        #########################################################################
        _, fashion_valid_h_z_np = get_dl_h_z(
            fashion_valid_mc_samples,
            mcd_samples_nro=cfg.mcd_runs,
            parallel_run=True
        )
        # Calculate entropy fashion test set
        _, fashion_test_h_z_np = get_dl_h_z(
            fashion_test_mc_samples,
            mcd_samples_nro=cfg.mcd_runs,
            parallel_run=True
        )
        # Calculate entropy ood test set
        _, ood_h_z_np = get_dl_h_z(
            mnist_test_mc_samples,
            mcd_samples_nro=cfg.mcd_runs,
            parallel_run=True
        )
        # Other entropy plots
        # plt.hist(fashion_valid_h_z_np.mean(axis=1), bins=200, label="InD valid")
        # plt.hist(fashion_test_h_z_np.mean(axis=1), bins=200, label="InD test")
        # plt.hist(ood_h_z_np.mean(axis=1), bins=200, label="OoD test")
        # plt.legend()
        #
        #
        # # In[32]:
        #
        #
        # plt.hist(fashion_valid_h_z_np.flatten(), bins=200, label="InD valid", alpha=0.5)
        # plt.hist(fashion_test_h_z_np.flatten(), bins=200, label="InD test", alpha=0.5)
        # plt.hist(ood_h_z_np.flatten(), bins=200, label="OoD test", alpha=0.5)
        # plt.legend()
        # plt.ylim((0,100000))
        #
        pacmap_2d_proj_plot = plot_samples_pacmap(samples_ind=fashion_test_h_z_np,
                                                  samples_ood=ood_h_z_np,
                                                  neighbors=10,
                                                  title="FashionMNIST - MNIST : $\hat{H}_{\phi}(z_i \mid x)$",
                                                  return_figure=True)
        mlflow.log_figure(figure=pacmap_2d_proj_plot,
                          artifact_file="figs/pacmap2d.png")
        #######################################################################
        # Start PCA evaluation
        #######################################################################
        pca_metrics = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                            'fpr', 'tpr', 'roc_thresholds',
                                            'precision', 'recall', 'pr_thresholds'])

        for n_components in tqdm(cfg.n_pca_components, desc="Evaluating PCA"):
            # Perform PCA dimension reduction
            pca_h_z_bdd_valid_samples, pca_transformation = apply_pca_ds_split(samples=fashion_valid_h_z_np,
                                                                               nro_components=n_components)
            pca_h_z_bdd_test_samples = apply_pca_transform(fashion_test_h_z_np, pca_transformation)
            pca_h_z_ood_samples = apply_pca_transform(ood_h_z_np, pca_transformation)

            # Build OoD detector
            scores_ind_test, scores_ood = fit_evaluate_KDE(h_z_ind_valid_samples=pca_h_z_bdd_valid_samples,
                                                           h_z_ind_test_samples=pca_h_z_bdd_test_samples,
                                                           h_z_ood_samples=pca_h_z_ood_samples,
                                                           normalize=False,
                                                           transform_exp=False)

            results_ood, results_for_mlflow = get_hz_detector_results(detect_exp_name=f"PCA_{n_components}_components",
                                                                      ind_samples_scores=scores_ind_test,
                                                                      ood_samples_scores=scores_ood,
                                                                      return_results_for_mlflow=True)
            results_for_mlflow = adjust_mlflow_results_name(data_dict=results_for_mlflow, technique_name='pca')
            mlflow.log_metrics(results_for_mlflow, step=n_components)
            pca_metrics = pca_metrics.append(results_ood)
        # Plot all PCA evaluations in one figure
        roc_curves_pca = save_roc_ood_detector(
            results_table=pca_metrics,
            plot_title=f"ROC Fashion MNIST vs MNIST OoD Detection PCA Conv layer"
        )
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
            pm_h_z_ind_valid_samples, pm_transformation = fit_pacmap(samples_ind=fashion_valid_h_z_np,
                                                                     neighbors=cfg.n_pacmap_neighbors,
                                                                     components=n_components)
            pm_h_z_ind_test_samples = apply_pacmap_transform(new_samples=fashion_test_h_z_np,
                                                             original_samples=fashion_valid_h_z_np,
                                                             pm_instance=pm_transformation)
            pm_h_z_ood_samples = apply_pacmap_transform(new_samples=ood_h_z_np,
                                                        original_samples=fashion_valid_h_z_np,
                                                        pm_instance=pm_transformation)

            # Build and evaluate OoD detector
            scores_ind_test, scores_ood = fit_evaluate_KDE(h_z_ind_valid_samples=pm_h_z_ind_valid_samples,
                                                           h_z_ind_test_samples=pm_h_z_ind_test_samples,
                                                           h_z_ood_samples=pm_h_z_ood_samples,
                                                           normalize=cfg.normalize_kde_prediction,
                                                           transform_exp=cfg.exp_kde)
            results_ood, results_for_mlflow = get_hz_detector_results(
                detect_exp_name=f"PacMAP_{n_components}_components",
                ind_samples_scores=scores_ind_test,
                ood_samples_scores=scores_ood,
                return_results_for_mlflow=True)
            results_for_mlflow = adjust_mlflow_results_name(data_dict=results_for_mlflow, technique_name='pm')

            mlflow.log_metrics(results_for_mlflow, step=n_components)
            pacmap_metrics = pacmap_metrics.append(results_ood)

        # Plot all PacMAP evaluations in one figure
        roc_curves_pacmap = save_roc_ood_detector(
            results_table=pacmap_metrics,
            plot_title=f"ROC Fashion MNIST vs MNIST OoD Detection PacMAP {cfg.layer_type}"
        )
        # Log the plot with mlflow
        mlflow.log_figure(figure=roc_curves_pacmap,
                          artifact_file="figs/roc_curves_pacmap.png")

    mlflow.end_run()


if __name__ == "__main__":
    main()
