import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

PREDICTIONS_CSV = os.path.join("datasets", "parallel_free_energy_predictions.csv")
OUTPUT_PLOT     = "plots/parallel_free_energy_avg_predictions.png"

def plot(start_index=0, smoothing_window=None):
    # Load predictions
    df = pd.read_csv(PREDICTIONS_CSV)

    # Drop 'chain' column if present
    if 'chain' in df.columns:
        df = df.drop(columns=['chain'])

    n_chains, n_windows = df.shape
    print(f"Loaded {n_chains} chains × {n_windows} windows")

    # Compute mean and standard error per window (column)
    means = df.mean(axis=0).values                          # (n_windows,)
    stds  = df.std(axis=0, ddof=1).values                   # (n_windows,)
    sems  = stds / np.sqrt(n_chains)                        # standard error of mean

    # X-axis: window indices
    x = np.arange(n_windows)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 6))

    # Mean line
    ax.plot(
        (x * 100)[start_index:], means[start_index:],
        color='steelblue', linewidth=2, label='Mean prediction', alpha=1
    )

    # ±2 SEM shaded region
    ax.fill_between(
        (x * 100)[start_index:],
        means[start_index:] - 2 * sems[start_index:],
        means[start_index:] + 2 * sems[start_index:],
        color='steelblue',
        alpha=0.3,
        label='±2 SEM'
    )

    # Smoothed line (covers entire domain)
    if smoothing_window is not None:
        smoothed = savgol_filter(means, window_length=smoothing_window, polyorder=3)
        ax.plot(
            (x * 100)[start_index:], smoothed[start_index:],
            color='red', linewidth=1,
            label=f'Savitzky-Golay Smoothing (window={smoothing_window})'
        )

    ax.set_xlabel('Starting frame per sliding window', fontsize=12)
    ax.set_ylabel(r'Average $\Delta F$ prediction', fontsize=12)
    ax.set_title(
        rf'Average $\Delta F$ Prediction Across {n_chains} Chains per Sliding Window',
        fontsize=14
    )
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_PLOT}")
    plt.show()


if __name__ == "__main__":
    plot(start_index=0, smoothing_window=None)