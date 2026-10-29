import os, re, warnings, shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Tuple

# ============================================================================
# CONFIG
# ============================================================================

Np = 20  # Max chain length (padding length)

PARAM_CONFIG = {
    'AB':    {'length': Np, 'type': 'vector'},
    'H':     {'length': Np, 'type': 'vector'},
    'K':     {'length': Np, 'type': 'vector'},
    'stdAB': {'length': Np, 'type': 'vector'},
    'stdH':  {'length': Np, 'type': 'vector'},
    'stdK':  {'length': Np, 'type': 'vector'},
    'alpha': {'length': 1,  'type': 'scalar'},
}

PARAM_NAMES = list(PARAM_CONFIG.keys())
TOTAL_FEATURES = sum(cfg['length'] for cfg in PARAM_CONFIG.values())

PARAM_SLICES = {}
_idx = 0
for _name, _cfg in PARAM_CONFIG.items():
    PARAM_SLICES[_name] = slice(_idx, _idx + _cfg['length'])
    _idx += _cfg['length']

# File-name tag for each parameter: "<AxBy>_<tag>.csv"
FEATURE_MAP = {
    'AB':    'meanEAB',
    'stdAB': 'stdEAB',
    'H':     'meanEH',
    'stdH':  'stdEH',
    'K':     'meanEK',
    'stdK':  'stdEK',
    'alpha': 'alpha',
}
TARGET_TAG = 'dF'

PARAM_LABELS = {
    'AB':    r'$\mathbb{E}[E_{AB}]$',
    'stdAB': r'$\sigma[E_{AB}]$',
    'H':     r'$\mathbb{E}[E_{H}]$',
    'stdH':  r'$\sigma[E_{H}]$',
    'K':     r'$\mathbb{E}[E_{K}]$',
    'stdK':  r'$\sigma[E_{K}]$',
    'alpha': r'$\alpha$',
}

CHAINS = [
    'A6B6', 'A9B3', 'A7B7', 'A9B5',
    'A8B8', 'A10B6', 'A9B9', 'A14B4',
]
DENSITIES = ['dens3', 'dens5']

# ----------------------------------------------------------------------------
# Roots — this script lives in github_upload/, datasets are right next to it.
# Resolve relative to the file so it works regardless of cwd.
# ----------------------------------------------------------------------------
HERE          = os.path.dirname(os.path.abspath(__file__))
DATASETS_ROOT = os.path.join(HERE, "datasets", "model_training")


# ============================================================================
# HELPERS
# ============================================================================

def _read_csv_rows(path: str) -> List[np.ndarray]:
    rows = []
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(np.fromstring(line, sep=',', dtype=np.float32))
    return rows

def build_valid_mask_full(chain_lengths: np.ndarray) -> np.ndarray:
    n_samples = len(chain_lengths)
    mask = np.ones((n_samples, TOTAL_FEATURES), dtype=bool)

    for name, cfg in PARAM_CONFIG.items():
        s = PARAM_SLICES[name]
        if cfg['type'] == 'vector':
            param_mask = np.zeros((n_samples, cfg['length']), dtype=bool)
            for i in range(n_samples):
                param_mask[i, :chain_lengths[i]] = True
            mask[:, s] = param_mask

    return mask

def build_valid_mask_param(param_name: str, chain_lengths: np.ndarray) -> np.ndarray:
    n_samples = len(chain_lengths)
    mask = np.zeros((n_samples, Np), dtype=bool)
    for i in range(n_samples):
        mask[i, :chain_lengths[i]] = True
    return mask

def _pad_rows(rows: List[np.ndarray], width: int = Np, fill: float = -1.0) -> np.ndarray:
    out = np.full((len(rows), width), fill, dtype=np.float32)
    for i, r in enumerate(rows):
        n = min(len(r), width)
        out[i, :n] = r[:n]
    return out


def infer_chain_lengths(features: np.ndarray) -> np.ndarray:
    ab_block = features[:, PARAM_SLICES['AB']]
    return np.sum(ab_block != -1.0, axis=1).astype(int)


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_chain_dataset(dens: str, chain: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one chain's full dataset from
        datasets/model_training/<dens>/<AxBy>_<tag>.csv
    for tag in {meanEAB, stdEAB, meanEH, stdEH, meanEK, stdEK, alpha, dF}.
    """
    folder = os.path.join(DATASETS_ROOT, dens)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing folder: {folder}")

    blocks: Dict[str, np.ndarray] = {}
    n_samples = None

    for name, cfg in PARAM_CONFIG.items():
        path = os.path.join(folder, f"{chain}_{FEATURE_MAP[name]}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {path}")

        rows = _read_csv_rows(path)
        if cfg['type'] == 'vector':
            block = _pad_rows(rows, width=cfg['length'], fill=-1.0)
        else:  # scalar (alpha)
            block = np.array([r[0] for r in rows], dtype=np.float32).reshape(-1, 1)

        if n_samples is None:
            n_samples = block.shape[0]
        elif block.shape[0] != n_samples:
            raise ValueError(
                f"Row count mismatch in {path}: {block.shape[0]} vs {n_samples}"
            )
        blocks[name] = block

    features = np.hstack([blocks[n] for n in PARAM_NAMES]).astype(np.float32)

    # Target
    target_path = os.path.join(folder, f"{chain}_{TARGET_TAG}.csv")
    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"Missing {target_path}")
    target_rows = _read_csv_rows(target_path)
    targets = np.array([r[0] for r in target_rows], dtype=np.float32).reshape(-1, 1)
    if targets.shape[0] != n_samples:
        raise ValueError(
            f"Target row count {targets.shape[0]} != feature row count {n_samples}"
        )

    chain_lengths = infer_chain_lengths(features)
    print(f"  Loaded {n_samples} samples: {dens}/{chain}")
    return features, targets, chain_lengths


def load_all_data(
    densities: List[str] = DENSITIES,
    chains: List[str] = CHAINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate all chains × densities into a single dataset."""
    feats, tgts, lens = [], [], []
    for dens in densities:
        for chain in chains:
            try:
                f, t, L = load_chain_dataset(dens, chain)
                feats.append(f); tgts.append(t); lens.append(L)
            except FileNotFoundError as e:
                print(f"  Skipping {dens}/{chain}: {e}")

    if not feats:
        raise RuntimeError("No data loaded.")

    features      = np.vstack(feats)
    targets       = np.vstack(tgts)
    chain_lengths = np.concatenate(lens)

    print(f"\n  Total samples: {len(features)}")
    print(f"  Chain length distribution: "
          f"{dict(zip(*np.unique(chain_lengths, return_counts=True)))}")
    return features, targets, chain_lengths

# ============================================================================
# 1. GRADIENT ANALYSIS — ACROSS ENTIRE DATASET
# ============================================================================

def compute_input_gradients(model: tf.keras.Model, features: np.ndarray,
                           batch_size: int = 1024) -> np.ndarray:
    all_grads = []
    n_samples = len(features)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = features[start:end]

        inputs_tensor = tf.Variable(batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            outputs = model(inputs_tensor, training=False)
            loss = tf.reduce_sum(outputs)
        grads = tape.gradient(loss, inputs_tensor).numpy()
        all_grads.append(grads)

    return np.vstack(all_grads)


def run_gradient_analysis(model: tf.keras.Model, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    print("\nLoading full dataset for gradient analysis...")
    features, targets, chain_lengths = load_all_data()
    n_samples = len(features)

    print(f"\nComputing gradients for {n_samples} samples...")
    gradients = compute_input_gradients(model, features)

    vector_params = [p for p in PARAM_NAMES if PARAM_CONFIG[p]['type'] == 'vector']

    # ---- Compute masked averages and standard deviations ----
    avg_grad = {}
    avg_abs_grad = {}
    std_grad = {}
    std_abs_grad = {}
    count_per_bead = {}

    for param in vector_params:
        s = PARAM_SLICES[param]
        param_grads = gradients[:, s]
        valid_mask = build_valid_mask_param(param, chain_lengths)

        masked_grads = np.where(valid_mask, param_grads, np.nan)
        masked_abs_grads = np.where(valid_mask, np.abs(param_grads), np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_grad[param] = np.nanmean(masked_grads, axis=0)
            avg_abs_grad[param] = np.nanmean(masked_abs_grads, axis=0)
            std_grad[param] = np.nanstd(masked_grads, axis=0, ddof=1)
            std_abs_grad[param] = np.nanstd(masked_abs_grads, axis=0, ddof=1)

        count_per_bead[param] = valid_mask.sum(axis=0)

    # Alpha scalar
    s_alpha = PARAM_SLICES['alpha']
    alpha_grads = gradients[:, s_alpha].flatten()
    alpha_avg_grad = np.mean(alpha_grads)
    alpha_avg_abs_grad = np.mean(np.abs(alpha_grads))
    alpha_std_abs_grad = np.std(np.abs(alpha_grads), ddof=1)

    # ---- Plot: Average Gradient per Bead with ±1σ error bars (2x3 grid) ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes_flat = axes.flatten()

    for i, param in enumerate(vector_params):
        ax = axes_flat[i]
        max_real = int(count_per_bead[param].nonzero()[0].max()) + 1 if count_per_bead[param].any() else Np
        beads = np.arange(max_real) + 1
        vals = avg_grad[param][:max_real]
        errs = std_grad[param][:max_real]

        ax.bar(beads, vals, yerr=errs, capsize=3, color='steelblue',
               alpha=0.7, edgecolor='black', linewidth=0.5,
               error_kw={'linewidth': 1.0, 'color': 'black'})
        ax.set_title(PARAM_LABELS[param], fontsize=18, fontweight='bold')
        ax.set_xlabel('Bead Index', fontsize=16)
        ax.set_ylabel('Average Gradient', fontsize=16)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(True, which='major', alpha=0.5)
        ax.grid(True, which='minor', alpha=0.2, linestyle=':')
        ax.axhline(0, color='k', linewidth=0.5)

    fig.suptitle(
        rf'Average Gradient per Bead with respect to...',
        fontsize=16, y=1.01
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "gradient_avg_full_dataset.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {path}")
    plt.show()

    # ---- Plot: Summary bar with ±1σ error bars ----
    fig, ax = plt.subplots(figsize=(10, 6))

    param_total_importance = []
    param_total_std = []
    labels_for_bar = []

    for param in PARAM_NAMES:
        if PARAM_CONFIG[param]['type'] == 'vector':
            valid_mask = build_valid_mask_param(param, chain_lengths)
            s = PARAM_SLICES[param]
            abs_grads = np.abs(gradients[:, s])

            # Per-sample: average |gradient| across valid beads only
            masked = np.where(valid_mask, abs_grads, np.nan)
            per_sample_mean = np.nanmean(masked, axis=1)

            param_total_importance.append(np.mean(per_sample_mean))
            param_total_std.append(np.std(per_sample_mean, ddof=1))
        else:
            param_total_importance.append(alpha_avg_abs_grad)
            param_total_std.append(alpha_std_abs_grad)
        labels_for_bar.append(PARAM_LABELS[param])

    sorted_idx = np.argsort(param_total_importance)[::-1]
    sorted_vals = [param_total_importance[i] for i in sorted_idx]
    sorted_errs = [param_total_std[i] for i in sorted_idx]
    sorted_labels = [labels_for_bar[i] for i in sorted_idx]

    ax.barh(range(len(PARAM_NAMES)), sorted_vals, xerr=sorted_errs,
            capsize=4, color='teal', alpha=0.7, edgecolor='black',
            error_kw={'linewidth': 1.0, 'color': 'black'})
    ax.set_yticks(range(len(PARAM_NAMES)))
    ax.set_yticklabels(sorted_labels, fontsize=18)
    ax.set_xlabel(r'Mean |Gradient| $\pm 1\sigma$',
                  fontsize=16)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(save_dir, "gradient_parameter_importance.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {path}")
    plt.show()

    # ---- Print Summary ----
    print(f"\n{'='*70}")
    print(f"GRADIENT SUMMARY — FULL DATASET ({n_samples} samples)")
    print(f"{'='*70}")
    print(f"  {'Parameter':<25} {'Mean |Grad|':<15} {'± 1σ':<15} {'Rank'}")
    print(f"  {'-'*65}")
    for rank, i in enumerate(sorted_idx):
        print(f"  {PARAM_NAMES[i]:<25} {param_total_importance[i]:<15.6f} "
              f"±{param_total_std[i]:<14.6f} #{rank+1}")
    print(f"{'='*70}\n")


# ============================================================================
# 2. SHAP ANALYSIS
# ============================================================================

def aggregate_shap_to_params_masked(shap_values_features: np.ndarray,
                                    chain_lengths: np.ndarray) -> np.ndarray:
    n_samples = len(shap_values_features)
    full_mask = build_valid_mask_full(chain_lengths)

    masked_shap = shap_values_features * full_mask

    param_shap = []
    for name in PARAM_NAMES:
        s = PARAM_SLICES[name]
        param_shap.append(masked_shap[:, s].sum(axis=1, keepdims=True))

    return np.hstack(param_shap)


def run_shap_full_dataset(model: tf.keras.Model, save_dir: str,
                          n_background: int = 100, n_explain: int = 300,
                          nsamples_kernel: int = 200):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*70)
    print("SHAP ANALYSIS — FULL DATASET (MASKED)")
    print("="*70)

    print("\nLoading full dataset for SHAP...")
    features, targets, chain_lengths = load_all_data()
    n_total = len(features)

    bg_idx = np.random.choice(n_total, min(n_background, n_total), replace=False)
    background = features[bg_idx].astype(np.float32)

    explain_idx = np.random.choice(n_total, min(n_explain, n_total), replace=False)
    explain_data = features[explain_idx].astype(np.float32)
    explain_chain_lengths = chain_lengths[explain_idx]

    print(f"  Background samples: {len(background)}")
    print(f"  Explain samples:    {len(explain_data)}")

    def predict_func(x):
        return model(x.astype(np.float32), training=False).numpy().flatten()

    print("\nInitializing KernelSHAP explainer...")
    explainer = shap.KernelExplainer(predict_func, background)

    print(f"Computing SHAP values (nsamples={nsamples_kernel})...")
    shap_values_features = explainer.shap_values(explain_data, nsamples=nsamples_kernel)

    if isinstance(shap_values_features, list):
        shap_values_features = shap_values_features[0]
    shap_values_features = np.array(shap_values_features)

    shap_values_params = aggregate_shap_to_params_masked(shap_values_features,
                                                         explain_chain_lengths)

    # Save raw values
    csv_path = os.path.join(save_dir, "shap_values_full_dataset.csv")
    np.savetxt(csv_path, shap_values_params, delimiter=",",
              header=",".join(PARAM_NAMES), comments="")
    np.save(os.path.join(save_dir, "shap_values_features_full.npy"), shap_values_features)
    np.save(os.path.join(save_dir, "shap_values_params_full.npy"), shap_values_params)
    np.save(os.path.join(save_dir, "explain_data_full.npy"), explain_data)
    np.save(os.path.join(save_dir, "explain_chain_lengths_full.npy"), explain_chain_lengths)
    print(f"  ✓ Saved raw SHAP data to {save_dir}/")

    # Terminal summary
    mean_abs = np.abs(shap_values_params).mean(axis=0)
    rank_order = np.argsort(-mean_abs)

    print(f"\n{'='*70}")
    print(f"SHAP SUMMARY — FULL DATASET ({len(explain_data)} explained samples)")
    print(f"{'='*70}")
    print(f"  {'Parameter':<25} {'Mean |SHAP|':<15} {'Mean SHAP':<15} {'Std SHAP':<15} {'Rank'}")
    print(f"  {'-'*75}")
    for rank, i in enumerate(rank_order):
        name = PARAM_NAMES[i]
        print(f"  {name:<25} {mean_abs[i]:<15.6f} "
              f"{shap_values_params[:, i].mean():<15.6f} "
              f"{shap_values_params[:, i].std():<15.6f} #{rank+1}")
    print(f"{'='*70}\n")

    # ---- Plot: SHAP beeswarm (full dataset) ----
    explain_data_params = np.zeros((len(explain_data), len(PARAM_NAMES)), dtype=np.float32)
    for j, name in enumerate(PARAM_NAMES):
        s = PARAM_SLICES[name]
        if PARAM_CONFIG[name]['type'] == 'vector':
            param_block = explain_data[:, s]
            valid_mask = np.zeros_like(param_block, dtype=bool)
            for i in range(len(explain_data)):
                valid_mask[i, :explain_chain_lengths[i]] = True
            masked_block = np.where(valid_mask, param_block, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                explain_data_params[:, j] = np.nanmean(masked_block, axis=1)
        else:
            explain_data_params[:, j] = explain_data[:, s].flatten()

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values_params, explain_data_params,
                     feature_names=[PARAM_LABELS[p] for p in PARAM_NAMES], show=False)
    # plt.title(f"SHAP Feature Impact (Testing Dataset)")
    plt.tight_layout()
    impact_path = os.path.join(save_dir, "shap_summary_impact_full_dataset.png")
    plt.savefig(impact_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved Impact Plot -> {impact_path}")

    return shap_values_params, shap_values_features

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    MODEL_PATH = "models/alpha_AB_H_K_std_dF_model/best_model.keras"

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.trainable = False
    print("Model loaded.\n")
    model.summary()

    print("\n" + "#"*70)
    print("# 1. GRADIENT ANALYSIS — FULL DATASET (±1σ)")
    print("#"*70)
    run_gradient_analysis(model, save_dir="plots/gradient_results")

    print("\n" + "#"*70)
    print("# 2. SHAP ANALYSIS — FULL DATASET (MASKED)")
    print("#"*70)
    run_shap_full_dataset(
        model, save_dir="plots/shap_results",
        n_background=100,
        n_explain=300,
        nsamples_kernel=200
    )

    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print("  Gradient plots & summary:  plots/gradient_results/")
    print("  SHAP plots & raw data:     plots/shap_results/")
    print("="*70)