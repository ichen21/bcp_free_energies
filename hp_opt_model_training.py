import os
import gc
import json
import numpy as np
import tensorflow as tf

# ============================
# Config
# ============================
DATA_DIR = "datasets/model_training"
FEATURE_NAMES = [
    "meanEAB", "stdEAB", "covEAB",
    "meanEH",  "stdEH",  "covEH",
    "meanEK",  "stdEK",  "covEK",
    "alpha",          # scalar feature; must stay last
]
TARGET_NAME = "dF"
Np = 20


# ============================
# Data pipeline
# ============================
def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def create_datasets(holdout_split=(0.1, 0.9), Np=20):
    def parse_line(*lines):
        *feature_lines, target_line = lines
        rows = []
        for line in feature_lines:
            row = tf.strings.to_number(tf.strings.split(line, ","), tf.float32)
            n = tf.shape(row)[0]
            rows.append(
                tf.cond(
                    n < 2,                                     # scalar (alpha)
                    lambda: row,
                    lambda: tf.pad(row, [[0, Np - n]], constant_values=-1),
                )
            )
        features = tf.concat(rows, axis=0)
        target   = tf.strings.to_number(tf.strings.split(target_line, ","), tf.float32)
        return features, target

    def make_ds(split):
        files = [os.path.join(DATA_DIR, f"{split}_{f}.csv") for f in FEATURE_NAMES]
        files.append(os.path.join(DATA_DIR, f"{split}_{TARGET_NAME}.csv"))
        line_dsets = tuple(tf.data.TextLineDataset(f) for f in files)
        return tf.data.Dataset.zip(line_dsets).map(
            parse_line, num_parallel_calls=tf.data.AUTOTUNE
        )

    train_n   = count_lines(os.path.join(DATA_DIR, f"training_{TARGET_NAME}.csv"))
    holdout_n = count_lines(os.path.join(DATA_DIR, f"holdout_{TARGET_NAME}.csv"))

    train_ds = (
        make_ds("training")
        .shuffle(train_n, reshuffle_each_iteration=True)
        .cache()
    )
    holdout_ds = make_ds("holdout").shuffle(holdout_n, reshuffle_each_iteration=False)

    n_val  = int(holdout_split[0] * holdout_n)
    n_test = int(holdout_split[1] * holdout_n)
    val_ds  = holdout_ds.take(n_val).cache()
    test_ds = holdout_ds.skip(n_val).take(n_test)

    return train_ds, val_ds, test_ds, train_n, holdout_n


# ============================
# Model
# ============================
def make_loss(name):
    return {
        "mse":      tf.keras.losses.MeanSquaredError(),
        "mae":      tf.keras.losses.MeanAbsoluteError(),
        "huber_1":  tf.keras.losses.Huber(delta=1.0),
        "huber_0_5":tf.keras.losses.Huber(delta=0.5),
    }[name.lower()]


def build_model_from_hp(hp, num_features):
    num_layers  = hp.Int("num_layers", 4, 12)
    nodes       = hp.Int("nodes_per_layer", 64, 1024, step=32)
    activation  = hp.Choice("activation", ["relu", "leaky_relu"])
    leaky_alpha = hp.Choice("leaky_alpha", [0.01, 0.1, 0.2, 0.3])
    lr          = hp.Choice("lr", [0.1, 0.01, 0.001, 0.0001])
    loss_name   = hp.Choice("loss", ["mse", "mae", "huber_1", "huber_0_5"])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(num_features,)))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(nodes))
        if activation == "relu":
            model.add(tf.keras.layers.ReLU())
        else:
            model.add(tf.keras.layers.LeakyReLU(alpha=leaky_alpha))
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=make_loss(loss_name),
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model


# ============================
# Prediction helpers
# ============================
def extract_predictions_and_targets(model, dataset, batch_size=2048):
    batched = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    targets = np.concatenate([y.numpy() for _, y in batched], axis=0).squeeze()
    preds   = model.predict(batched, verbose=0).squeeze()
    return preds, targets


def save_parity_data(preds, targets, out_dir, prefix):
    np.save(os.path.join(out_dir, f"{prefix}_predictions.npy"), preds)
    np.save(os.path.join(out_dir, f"{prefix}_targets.npy"), targets)
    csv_path = os.path.join(out_dir, f"{prefix}_parity.csv")
    with open(csv_path, "w") as f:
        f.write("true,predicted\n")
        for t, p in zip(targets, preds):
            f.write(f"{t},{p}\n")
    print(f"  Saved {prefix} parity data ({len(targets)} points) -> {csv_path}")


# ============================
# Main
# ============================
if __name__ == "__main__":
    eval_batch_size  = 2048
    holdout_val_frac = 0.10
    epochs           = 40
    patience         = 10
    max_trials       = 40
    executions_per_trial = 1

    tuner_dir    = "models"
    project_name = "alpha_AB_H_K_std_cov_dF_model_hpo"

    train_ds_raw, val_ds_raw, test_ds_raw, train_n, holdout_n = create_datasets(
        holdout_split=(holdout_val_frac, 1.0 - holdout_val_frac), Np=Np
    )

    val_ds_eval  = val_ds_raw.batch(eval_batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds_eval = test_ds_raw.batch(eval_batch_size).prefetch(tf.data.AUTOTUNE)

    # 9 padded vector features + 1 scalar (alpha)
    num_features = (len(FEATURE_NAMES) - 1) * Np + 1

    val_num   = int(holdout_n * holdout_val_frac)
    val_steps = max(1, int(np.ceil(val_num / eval_batch_size)))

    print(f"train_n = {train_n}, holdout_n = {holdout_n}")
    print(f"num_features = {num_features}, val_steps = {val_steps}")

    try:
        import keras_tuner as kt
    except ImportError as e:
        raise ImportError("keras-tuner is required: pip install keras-tuner") from e

    class BatchSizeTuner(kt.RandomSearch):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            batch_size = hp.Choice("batch_size", values=[32, 64, 128, 256, 512])
            train_batched = train_ds_raw.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            kwargs["x"]                = train_batched
            kwargs["validation_data"]  = val_ds_eval
            kwargs["steps_per_epoch"]  = int(np.round(train_n / batch_size))
            kwargs["validation_steps"] = val_steps
            return super().run_trial(trial, *args, **kwargs)

    tuner = BatchSizeTuner(
        hypermodel=lambda hp: build_model_from_hp(hp, num_features=num_features),
        objective=kt.Objective("val_mse", direction="min"),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=tuner_dir,
        project_name=project_name,
        overwrite=False,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    tuner.search(epochs=epochs, callbacks=[early_stopping], verbose=1)

    # ============================
    # Export best results
    # ============================
    best_hp    = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    print("\nBEST HP:")
    print(best_hp.values)

    test_metrics = best_model.evaluate(test_ds_eval, verbose=0)
    test_report  = dict(zip(best_model.metrics_names, [float(x) for x in test_metrics]))

    print("\nHOLDOUT TEST METRICS:")
    for k, v in test_report.items():
        print(f"  {k}: {v}")

    out_dir = os.path.join(tuner_dir, project_name, "best_export")
    os.makedirs(out_dir, exist_ok=True)

    best_model_path = os.path.join(out_dir, "best_model.keras")
    best_model.save(best_model_path)
    with open(os.path.join(out_dir, "best_hparams.json"), "w") as f:
        json.dump(best_hp.values, f, indent=2)
    with open(os.path.join(out_dir, "test_report.json"), "w") as f:
        json.dump(test_report, f, indent=2)

    # ============================
    # Parity data
    # ============================
    parity_dir = os.path.join(out_dir, "parity_data")
    os.makedirs(parity_dir, exist_ok=True)

    print("\nExtracting training-set predictions ...")
    train_preds, train_targets = extract_predictions_and_targets(
        best_model, train_ds_raw, batch_size=eval_batch_size
    )
    save_parity_data(train_preds, train_targets, parity_dir, prefix="train")

    print("Extracting test-set predictions ...")
    test_preds, test_targets = extract_predictions_and_targets(
        best_model, test_ds_raw, batch_size=eval_batch_size
    )
    save_parity_data(test_preds, test_targets, parity_dir, prefix="test")

    summary = {
        "train_n_samples": int(len(train_targets)),
        "train_mse": float(np.mean((train_preds - train_targets) ** 2)),
        "train_mae": float(np.mean(np.abs(train_preds - train_targets))),
        "test_n_samples": int(len(test_targets)),
        "test_mse": float(np.mean((test_preds - test_targets) ** 2)),
        "test_mae": float(np.mean(np.abs(test_preds - test_targets))),
    }
    with open(os.path.join(parity_dir, "parity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nParity summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"All parity files saved to: {parity_dir}")

    # ============================
    # Trials CSV
    # ============================
    results = []
    for t in tuner.oracle.get_best_trials(num_trials=max_trials):
        row = {"trial_id": t.trial_id, "score_val_loss": float(t.score)}
        row.update(t.hyperparameters.values)
        results.append(row)
    results.sort(key=lambda r: r["score_val_loss"])

    csv_path = os.path.join(out_dir, "trials_sorted.csv")
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in results:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    print(f"\nSaved best model to: {best_model_path}")
    print(f"Saved trials to: {csv_path}")

    gc.collect()