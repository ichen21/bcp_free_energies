import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from utils import csv_to_array
import pandas as pd
from sklearn.metrics import r2_score
import os

def _plot_single_parity(ax, train_csv, test_csv, text, gridsize=60):
    # --- Load data ---
    test_df = pd.read_csv(test_csv)
    y_pred = test_df["predicted"].values
    y_true = test_df["true"].values
    print(f"{text} -> Test MSE: {np.mean((y_pred - y_true) ** 2):.6g}")

    train_df = pd.read_csv(train_csv)
    y_pred_tr = train_df["predicted"].values
    y_true_tr = train_df["true"].values

    # --- Metrics ---
    test_R2 = r2_score(y_true, y_pred)
    train_R2 = r2_score(y_true_tr, y_pred_tr)

    # --- Fit (test only) ---
    coef = np.polyfit(y_true, y_pred, 1)
    f = np.poly1d(coef)

    # --- Axis range ---
    all_vals = np.concatenate([y_true, y_pred, y_true_tr, y_pred_tr])
    low, high = np.floor(all_vals.min()), np.ceil(all_vals.max())
    xline = np.linspace(low, high, 200)

    # --- Train background ---
    ax.hexbin(
        y_true_tr, y_pred_tr,
        gridsize=gridsize,
        mincnt=1,
        norm=LogNorm(),
        cmap="Blues",
        alpha=0.6
    )

    # --- Test overlay ---
    ax.scatter(
        y_true, y_pred,
        facecolors='none',
        edgecolors='red',
        s=12,
        linewidths=0.6,
        alpha=0.5
    )

    # --- Lines ---
    ax.plot(xline, f(xline), '--', color='red', linewidth=1)
    ax.plot(xline, xline, '--', color='black', linewidth=1)

    # --- Labels ---
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal', adjustable='box')

    ax.set_title(rf"Train $R^2 =$ {train_R2:.4g} | Test $R^2 =$ {test_R2:.4g}", fontsize=12)

    ax.text(
        0.05, 0.9, text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )


def plot_model_grid(base_dir="models", save_path="plots/all_models_parity_grid.png"):
    """
    Plots the 2x2 grid of parity plots for the four progressively augmented
    statistical-moment models:
        (i)   alpha only
        (ii)  alpha, {E[.]}
        (iii) alpha, {E[.]}, {sigma[.]}
        (iv)  alpha, {E[.]}, {sigma[.]}, {Cov[., .]}
    """
    models = [
        # Row 1
        ("alpha_dg_model", r"(i) $\alpha$ only"),
        ("alpha_AB_H_K_dF_model", r"(ii) $\alpha$, $\{\mathbb{E}[\cdot]\}$"),

        # Row 2
        ("alpha_AB_H_K_std_dF_model", r"(iii) $\alpha$, $\{\mathbb{E}[\cdot]\}, \{\sigma[\cdot]\}$"),
        ("alpha_AB_H_K_std_cov_dF_model", r"(iv) $\alpha$, $\{\mathbb{E}[\cdot]\}, \{\sigma[\cdot]\}, \{\mathrm{Cov}[\cdot, \cdot]\}$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))

    for ax, (folder, text) in zip(axes.flatten(), models):
        parity_dir = os.path.join(base_dir, folder, "parity_data")
        train_csv = os.path.join(parity_dir, "train_parity.csv")
        test_csv  = os.path.join(parity_dir, "test_parity.csv")
        _plot_single_parity(ax, train_csv, test_csv, text)

    # Axis labels only on outer edges
    for ax in axes[-1, :]:
        ax.set_xlabel("True dF", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted dF", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved parity grid to {save_path}")


if __name__ == "__main__":
    plot_model_grid(
        base_dir="models",
        save_path="plots/all_models_parity_grid.png"
    )
