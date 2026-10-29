import numpy as np
from utils import csv_to_array
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

plt.rcParams.update({
    "font.size": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# Define datasets
datasets = [
    {"dir": "dens3/A9B9", "z": 0.2837, "N": 18},
    {"dir": "dens3/A14B4", "z": 0.2837, "N": 18},
    {"dir": "dens5/A9B9", "z": 0.6607, "N": 18},
    {"dir": "dens5/A14B4", "z": 0.6607, "N": 18},
]

# Load model
model = tf.keras.models.load_model("models/alpha_AB_H_K_std_dF_model/best_model.keras")

def load_model_predictions(dens_str, chain):
    """Load model predictions for given alphas"""
    predictions = []
    
    features_file = f"datasets/model_training/{dens_str}/{chain}_{{}}.csv"
    
    feature_names = ["meanEAB", "stdEAB", "meanEH", "stdEH", "meanEK", "stdEK", "alpha"]
    
    input_vectors = None

    for feat in feature_names:
        feat_data = csv_to_array(features_file.format(feat))
        padded_data = np.concatenate([feat_data, np.full((feat_data.shape[0], 2), -1)], axis=1) if feat != "alpha" else feat_data
        # print(padded_data.shape)
        if input_vectors is None:
            input_vectors = padded_data
        else:
            input_vectors = np.hstack([input_vectors, padded_data])
    
    # Get prediction
    return model.predict(input_vectors, verbose=0)
    

def load_bf_predictions(dens_str, chain, alphas, z, N):
    """Load brute force dF predictions"""
    bf_predictions = []
    
    for alpha in alphas:
        chi_n = z * alpha * N
        bf_file = f"datasets/brute_force_dF/{dens_str}_{chain}/{alpha}_BFdG.csv"
        
        try:
            bf_data = csv_to_array(bf_file)
            bf_predictions.append(bf_data.mean())
        except FileNotFoundError:
            bf_predictions.append(np.nan)
    
    return np.array(bf_predictions)

# Plot for each dataset
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, dataset in enumerate(datasets):
    dens_dir = dataset["dir"]
    z = dataset["z"]
    N = dataset["N"]
    dens_str = dens_dir.split("/")[0]  # "dens3" or "dens5"
    chain = dens_dir.split("/")[1]  # "A9B9" or "A14B4"
    
    # Load alphas from file
    alpha_file = f"datasets/model_training/{dens_str}/{chain}_alpha.csv"
    alphas = csv_to_array(alpha_file).flatten()
    
    # Convert alphas to chiN
    chi_N = z * alphas * N
    
    # Load true dF
    true_dF_file = f"datasets/model_training/{dens_str}/{chain}_dF.csv"
    true_dF = csv_to_array(true_dF_file).flatten()
    
    # Load BF dF
    bf_dF = load_bf_predictions(dens_str, chain, alphas, z, N)
    
    # Load model dF
    model_dF = load_model_predictions(dens_str, chain)
    
    # Sort by chiN for plotting
    sort_idx = np.argsort(chi_N)
    chi_N_sorted = chi_N[sort_idx]
    true_dF_sorted = true_dF[sort_idx]
    bf_dF_sorted = bf_dF[sort_idx]
    model_dF_sorted = model_dF[sort_idx]
    
    # Plot
    ax = axes[idx]
    
    ax.set_title(rf"$\Delta F$ vs. $\chi N$ for {chain}, $\rho = {dens_str.replace('dens', '')}$")
    ax.set_xlabel(r"$\chi N$")
    ax.set_ylabel(r"$\Delta F$")
    
    ax.plot(
        chi_N_sorted, true_dF_sorted,
        marker='o', linestyle='--', alpha=0.7, markersize=6, color="black",
        label=r"true $\Delta F$"
    )
    
    ax.plot(
        chi_N_sorted, bf_dF_sorted,
        marker='x', linestyle='', alpha=0.7, markersize=8, color="red",
        label=r"brute force $\Delta F$"
    )
    
    ax.plot(
        chi_N_sorted, model_dF_sorted,
        marker='s', linestyle='', alpha=0.7, markersize=6, color="blue",
        label=r"model $\Delta F$"
    )
    
    ax.legend()
    ax.set_ylim(-0.5, 7.0)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/free_energy_comparison_all_datasets.png", dpi=300)
# plt.show()

print("Plotting complete!")