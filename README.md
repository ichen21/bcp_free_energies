# Rapid neural network prediction of block copolymer free energies

This repository contains the datasets, trained neural network models, and analysis scripts associated with the manuscript *Rapid neural network prediction of block copolymer free energies*. The provided code reproduces the model training, free-energy prediction, interpretability analyses, and figures presented in the paper.

## Datasets

The `datasets/` directory contains all simulation derived inputs and reference free energies used for model training, validation, testing, and downstream analysis. If the raw simulation data is needed, please contact us (see below) for the relevant files.

### `datasets/model_training/` (Hosted on Zenodo)

Due to file size limits, the model training dataset is hosted on Zenodo at [https://zenodo.org/records/23749838](https://zenodo.org/records/23749838). Before running the training or analysis scripts, download and extract the archive into `datasets/model_training/`. Once extracted, it contains the per-chain energetic descriptors and reference free-energy targets generated from dissipative particle dynamics (DPD) simulations of freely jointed chain block copolymers:

* `dens3/` and `dens5/`: per-chain datasets at bead number densities $$\rho = 3$$ and $$\rho = 5$$, respectively. For each chain architecture (e.g. `A9B9`, `A14B4`), the following CSV files are provided:
  * `{chain}_meanEAB.csv`, `{chain}_stdEAB.csv`: per-bead mean and standard deviation of the heterogeneous (A-B) interaction energy.
  * `{chain}_meanEH.csv`, `{chain}_stdEH.csv`: per-bead mean and standard deviation of the homogeneous (A-A, B-B) interaction energy.
  * `{chain}_meanEK.csv`, `{chain}_stdEK.csv`: per-bead mean and standard deviation of the bonded spring energy.
  * `{chain}_covEAB.csv`, `{chain}_covEH.csv`, `{chain}_covEK.csv`: bonded-pair covariances along the chain backbone for each energy component.
  * `{chain}_alpha.csv`: scalar DPD repulsion parameter $$\alpha$$ for each sample.
  * `{chain}_dF.csv`: reference excess free energy $$\Delta F$$ relative to $$\chi N = 0$$, computed via the stratified Bennett Acceptance Ratio (BAR) procedure.

* `training_{feature}.csv` and `holdout_{feature}.csv`: compiled training and holdout datasets aggregated across all chain architectures listed in Table 1 of the manuscript. The holdout set is further split internally (10 % validation, 90 % testing) by the training scripts. The chain architectures included in each split follow the partitioning described in the paper.

### `datasets/brute_force_dF/`

Pre-computed brute-force BAR free-energy estimates for the four chain architectures used in the comparison study (Section 4.3 of the manuscript). Each subdirectory contains files of the form `{alpha}_BFdG.csv`, where `{alpha}` is the repulsion parameter. Subdirectories:

* `dens3_A9B9/`, `dens3_A14B4/`: brute-force estimates at $$\rho = 3$$.
* `dens5_A9B9/`, `dens5_A14B4/`: brute-force estimates at $$\rho = 5$$.

### `datasets/parallel_free_energy_predictions.csv`

Pre-computed sliding-window $$\Delta F$$ predictions for the morphological transition experiment described in Section 4.4. Rows correspond to individual polymer chains and columns correspond to sliding-window starting frames sampled every 100 simulation time steps. The trajectory captures an equilibrated $$A_{10}B_{10}$$ lamellar system that is instantaneously converted to $$A_{14}B_6$$ at frame $5\times 10^5$ and subsequently relaxes through a metastable perforated lamellar intermediate to the equilibrium cylindrical morphology.

## Models

The `models/` directory contains the four trained feed-forward neural networks compared in the manuscript, each corresponding to a progressively richer statistical-moment input-feature set as defined in Table 4. The naming convention encodes the included features:

| Directory | Manuscript label | Input features |
| --- | --- | --- |
| `alpha_dg_model/` | (i) | $$\alpha$$ |
| `alpha_AB_H_K_dF_model/` | (ii) | $$\alpha$$, $$\{\mathbb{E}[\cdot]\}$$ |
| `alpha_AB_H_K_std_dF_model/` | (iii) | $$\alpha$$, $$\{\mathbb{E}[\cdot]\}$$, $$\{\sigma[\cdot]\}$$ |
| `alpha_AB_H_K_std_cov_dF_model/` | (iv) | $$\alpha$$, $$\{\mathbb{E}[\cdot]\}$$, $$\{\sigma[\cdot]\}$$, $$\{\mathrm{Cov}[\cdot,\cdot]\}$$ |

Each model directory contains:

* `best_model.keras`: trained Keras model file. Use this artifact for inference.
* `best_hparams.json`: optimized hyperparameters identified by the random search.
* `test_report.json`: evaluation metrics (MSE, MAE) on the held-out test set.
* `trials_sorted.csv`: full record of hyperparameter search trials, sorted by validation loss.
* `parity_data/`: precomputed predictions on the training and test sets, including `train_parity.csv`, `test_parity.csv`, and `parity_summary.json`.

Model (iii), located at `models/alpha_AB_H_K_std_dF_model/best_model.keras`, is the recommended model for inference and is used by all downstream analysis scripts.

## Plots

The `plots/` directory stores figures produced by the analysis scripts, including the energy autocovariance/autocorrelation plots, the parity-plot grid, the brute-force comparison plots, the gradient and SHAP interpretability plots, and the parallel free-energy prediction time series.

## Python scripts

All scripts assume execution from the repository root and expect the directory layout above. Required packages: `tensorflow`, `keras-tuner`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, and `shap`.

### `hp_opt_model_training.py`

Trains a feed-forward neural network on a specified input-feature set using random-search hyperparameter optimization (`kt.RandomSearch`) via `keras-tuner`. Variable-length per-bead feature vectors are right-padded to length 20 with a sentinel value of `-1` and passed directly into the dense network layers. The script streams the compiled training and holdout CSVs from `datasets/model_training/`, and searches over the number of hidden layers, nodes per layer, activation function, learning rate, loss function, and minibatch size (Table 3 of the manuscript). The best model, hyperparameters, test-set metrics, and training/test parity data are exported to `models/<project_name>/best_export/`. The active feature set is controlled by the `FEATURE_NAMES` list and the `project_name` variable at the top of the script.

### `all_models_parity_plots.py`

Generates the 2x2 grid of parity plots comparing predicted and reference $$\Delta F$$ for all four models (Figure 4 of the manuscript). For each model, training-set predictions are rendered as a logarithmic hexbin density and test-set predictions are overlaid as scatter points. Test and train $$R^2$$ values are annotated on each subplot. The script reads `train_parity.csv` and `test_parity.csv` from each model's `parity_data/` subdirectory and writes the combined figure to `plots/all_models_parity_grid.png`.

### `compare_free_energy_outputs.py`

Reproduces the comparison between the trained neural network (model iii), the brute-force BAR estimator, and the stratified BAR reference free energies for $$A_9B_9$$ and $$A_{14}B_4$$ at $$\rho = 3$$ and $$\rho = 5$$ (Figure 9). For each system, the script loads the per-chain feature vectors, runs inference using `models/alpha_AB_H_K_std_dF_model/best_model.keras`, loads the corresponding brute-force estimates from `datasets/brute_force_dF/`, and overlays all three curves on a $$\Delta F$$ versus $$\chi N$$ axis. Output figure: `plots/free_energy_comparison_all_datasets.png`.

### `gradient_shap_computation.py`

Computes interpretability diagnostics for the trained model (iii), corresponding to Figures 7 and 8 of the manuscript. The script performs two analyses:

1. **Gradient analysis**: per-bead input gradients $$\partial \Delta F / \partial x_i$$ are evaluated across the full test dataset. Per-bead averages with $$\pm 1\sigma$$ error bars are reported for each vector feature, and parameter-level mean absolute gradients are summarized in a horizontal bar plot. Padded input positions (`-1` sentinel values at $$i > N$$) are excluded in post-processing so that only valid bead positions contribute to the statistics.

2. **SHAP analysis**: KernelSHAP values are computed on the 121-dimensional input space for a subsample of the test dataset (default 100 background points, 300 explanation points, 200 coalition evaluations). Padded positions ($$i > N$$) are zeroed out and valid per-bead SHAP values ($$1 \leq i \leq N$$) are summed within each parameter class via Shapley additivity to obtain the net attribution per feature class, with beeswarm points colored by the mean feature value across valid beads.

Outputs are saved to `plots/gradient_results/` and `plots/shap_results/`, including raw SHAP arrays in `.npy` and `.csv` formats for downstream reuse.

### `parallel_dF_prediction_plot.py`

Plots the time series of the system-averaged $\Delta F$ prediction for the lamellae-to-cylinder transition described in Section 4.4 (Figure 11). The script loads the precomputed sliding-window predictions from `datasets/parallel_free_energy_predictions.csv`, computes the mean and standard error of the mean across all chains for each window, and renders the result as a line plot with a $\pm 2$ SEM shaded confidence band. An optional Savitzky-Golay smoothing overlay can be enabled via the `smoothing_window` argument of the `plot()` function. Output figure: `plots/parallel_free_energy_avg_predictions.png`.

## Reproducing the figures

Ensure that the model training dataset from Zenodo ([https://zenodo.org/records/23749838](https://zenodo.org/records/23749838)) has been downloaded and extracted into `datasets/model_training/` first. The figures in the manuscript can then be regenerated from the provided artifacts as follows:

* Figure 4 (parity grid): `python all_models_parity_plots.py`
* Figures 7 and 8 (gradient and SHAP interpretability): `python gradient_shap_computation.py`
* Figure 9 (model versus brute-force BAR comparison): `python compare_free_energy_outputs.py`
* Figure 11 (sliding-window free-energy time series): `python parallel_dF_prediction_plot.py`

Retraining a model from scratch is performed with `python hp_opt_model_training.py` after editing the `FEATURE_NAMES` list and `project_name` at the top of the script to select the desired input-feature set.

## Citation

If you use this dataset or code, please cite the associated manuscript:

> Chen, Ian, and Alfredo Alexander-Katz. "Rapid Neural Network Prediction of Linear Block Copolymer Free Energies." arXiv preprint arXiv:2603.17391 (2026).

## Contact

For questions regarding the data or code, please contact Ian Chen (`ischen@mit.edu`) or Alfredo Alexander-Katz (`aalexand@mit.edu`).
