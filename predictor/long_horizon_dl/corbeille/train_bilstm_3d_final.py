"""
Train the HGW Bi-LSTM 3-day model.

This script is designed for the project demo workflow:
- keep the Bi-LSTM long-horizon model in the system;
- train or retrain it when you have time;
- save artifacts directly where prediction code already loads them;
- back up existing artifacts before replacing them.
"""

import argparse
import json
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "hgw_5yr_bigdata.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR

GRANULARITY_MIN = 30
LOOKBACK_HOURS = 24
HORIZON_HOURS = 72
SUBSAMPLE = 2
LOOKBACK_STEPS = LOOKBACK_HOURS * 60 // GRANULARITY_MIN
SEQ_LEN = LOOKBACK_STEPS // SUBSAMPLE

LABEL_COL = "incident_in_72h"
INCIDENT_STATUSES = {"CRITICAL", "URGENT"}

SHARED_FEATURES = [
    "cpu_load",
    "mem_used_pct",
    "ping_latency",
    "packet_loss",
    "wan_status",
    "cpu_mean_24h",
    "ram_mean_24h",
    "cpu_std_24h",
    "ram_std_24h",
    "cpu_slope_6h",
    "ram_slope_6h",
    "wan_instability_6h",
    "health_score",
]

ARTIFACTS = [
    "bilstm_3d.weights.h5",
    "bilstm_3d_synthetic.keras",
    "transfer_scaler.pkl",
    "bilstm_3d_metadata.json",
]


def compute_health_score(cpu, mem, ping, loss):
    cpu, mem, ping, loss = [np.asarray(x, dtype=float) for x in [cpu, mem, ping, loss]]
    n_cpu = np.clip((cpu - 20) / 70, 0, 1)
    n_mem = np.clip((mem - 35) / 55, 0, 1)
    n_ping = np.clip((ping - 20) / 200, 0, 1)
    n_loss = np.clip(loss / 15, 0, 1)
    composite = 0.35 * n_mem + 0.30 * n_cpu + 0.20 * n_ping + 0.15 * n_loss
    return np.round((1.0 - np.clip(composite, 0, 1)) * 100, 1)


def build_bilstm(seq_len=SEQ_LEN, n_features=len(SHARED_FEATURES)):
    """Architecture intentionally matching bilstm_loader.py."""
    inp = Input(shape=(seq_len, n_features), name="input")
    x = Bidirectional(LSTM(48, return_sequences=True, dropout=0.3), name="bilstm_1")(inp)
    x = Bidirectional(LSTM(24, return_sequences=False, dropout=0.3), name="bilstm_2")(x)
    x = Dense(32, activation="relu", name="dense_1")(x)
    x = Dropout(0.3, name="dropout_1")(x)
    x = Dense(16, activation="relu", name="dense_2")(x)
    x = Dropout(0.2, name="dropout_2")(x)
    out = Dense(1, activation="sigmoid", name="prediction")(x)
    return Model(inputs=inp, outputs=out, name="bilstm_3day")


def parse_args():
    parser = argparse.ArgumentParser(description="Train HGW Bi-LSTM 3-day model")
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Training CSV. Can be engineered synthetic data or raw monitor snapshots.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory where bilstm artifacts are saved. Default: predictor/long_horizon_dl.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=2, help="Sequence stride inside each split.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--force-rebuild-label",
        action="store_true",
        help="For raw LOCAL_STATUS CSVs, rebuild incident_in_72h from CRITICAL/URGENT.",
    )
    parser.add_argument(
        "--keep-current-incidents",
        action="store_true",
        help="For raw CSVs, keep rows already in CRITICAL/URGENT. Default drops them for predictive training.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not copy existing artifacts to a timestamped backup folder before saving.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data, build splits/sequences, then stop before training.",
    )
    return parser.parse_args()


def load_training_frame(csv_path, force_rebuild_label=False, keep_current_incidents=False):
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a timestamp column")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    has_engineered = set(SHARED_FEATURES + [LABEL_COL]).issubset(df.columns)
    if has_engineered and not force_rebuild_label:
        keep_cols = ["timestamp"] + SHARED_FEATURES + [LABEL_COL]
        for col in ["gateway_id", "session_id"]:
            if col in df.columns:
                keep_cols.insert(1, col)
                break
        out = df[keep_cols].copy()
        out = out.dropna(subset=SHARED_FEATURES + [LABEL_COL]).reset_index(drop=True)
        out[LABEL_COL] = out[LABEL_COL].astype(int)
        return out, "engineered"

    raw_required = {
        "CPU_USAGE_PERCENT",
        "MEM_USAGE_PERCENT",
        "NET_LATENCY_MS",
        "NET_PING_STATUS",
        "WAN_STATE",
        "LOCAL_STATUS",
    }
    if not raw_required.issubset(df.columns):
        missing = sorted((set(SHARED_FEATURES + [LABEL_COL]) - set(df.columns)) | (raw_required - set(df.columns)))
        raise ValueError(f"CSV schema not supported. Missing columns include: {missing}")

    groups = [("", df)]
    group_col = None
    if "gateway_id" in df.columns:
        group_col = "gateway_id"
        groups = df.groupby(group_col, sort=False)

    prepared = []
    for group_id, group in groups:
        one = prepare_raw_group(group, group_id if group_col else None, group_col)
        prepared.append(one)

    out = pd.concat(prepared, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if not keep_current_incidents:
        out = out[out["current_incident"] == 0].reset_index(drop=True)
    out = out.dropna(subset=SHARED_FEATURES + [LABEL_COL]).reset_index(drop=True)
    out[LABEL_COL] = out[LABEL_COL].astype(int)
    return out, "raw_rebuilt"


def prepare_raw_group(group, group_id=None, group_col=None):
    g = group.copy().sort_values("timestamp")
    idx = pd.to_datetime(g["timestamp"])

    base = pd.DataFrame(index=idx)
    base["cpu_load"] = pd.to_numeric(g["CPU_USAGE_PERCENT"], errors="coerce").to_numpy()
    base["mem_used_pct"] = pd.to_numeric(g["MEM_USAGE_PERCENT"], errors="coerce").to_numpy()
    base["ping_latency"] = pd.to_numeric(g["NET_LATENCY_MS"], errors="coerce").to_numpy()
    base["packet_loss"] = (
        (g["NET_PING_STATUS"].astype(str).str.upper() == "FAIL").astype(float) * 100.0
    ).to_numpy()
    base["wan_status"] = (g["WAN_STATE"].astype(str).str.upper() == "UP").astype(float).to_numpy()

    data_30 = base.resample("30min").mean().ffill().dropna()
    data_30["ping_latency"] = data_30["ping_latency"].ffill().fillna(50.0)

    incident = g["LOCAL_STATUS"].astype(str).str.upper().isin(INCIDENT_STATUSES).astype(int)
    incident.index = idx
    incident_30 = incident.resample("30min").max().ffill().reindex(data_30.index).fillna(0).astype(int)

    win_24h = LOOKBACK_STEPS
    win_6h = 6 * 60 // GRANULARITY_MIN
    data_30["cpu_mean_24h"] = data_30["cpu_load"].rolling(win_24h, min_periods=1).mean()
    data_30["ram_mean_24h"] = data_30["mem_used_pct"].rolling(win_24h, min_periods=1).mean()
    data_30["cpu_std_24h"] = data_30["cpu_load"].rolling(win_24h, min_periods=1).std().fillna(0)
    data_30["ram_std_24h"] = data_30["mem_used_pct"].rolling(win_24h, min_periods=1).std().fillna(0)
    data_30["cpu_slope_6h"] = (data_30["cpu_load"] - data_30["cpu_load"].shift(win_6h)).fillna(0) / 6
    data_30["ram_slope_6h"] = (data_30["mem_used_pct"] - data_30["mem_used_pct"].shift(win_6h)).fillna(0) / 6
    data_30["wan_instability_6h"] = data_30["wan_status"].eq(0).rolling(win_6h, min_periods=1).mean()
    data_30["health_score"] = compute_health_score(
        data_30["cpu_load"].fillna(0),
        data_30["mem_used_pct"].fillna(0),
        data_30["ping_latency"].fillna(50),
        data_30["packet_loss"].fillna(0),
    )

    horizon_steps = HORIZON_HOURS * 60 // GRANULARITY_MIN
    future_incident = incident_30.iloc[::-1].rolling(horizon_steps, min_periods=1).max().iloc[::-1]
    data_30[LABEL_COL] = future_incident.shift(-1).fillna(0).astype(int).values
    data_30["current_incident"] = incident_30.values

    out = data_30.reset_index(names="timestamp")
    if group_col:
        out[group_col] = group_id
    return out


def temporal_split(df, train_ratio, val_ratio):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


def build_sequences(df_part, scaler, label_col=LABEL_COL, stride=2):
    if df_part.empty:
        return np.empty((0, SEQ_LEN, len(SHARED_FEATURES)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    group_col = "gateway_id" if "gateway_id" in df_part.columns else "session_id" if "session_id" in df_part.columns else None
    groups = df_part.groupby(group_col, sort=False) if group_col else [(None, df_part)]

    xs, ys = [], []
    for _, group in groups:
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) <= LOOKBACK_STEPS:
            continue
        x_scaled = scaler.transform(group[SHARED_FEATURES].values).astype(np.float32)
        labels = group[label_col].values.astype(np.float32)
        for i in range(LOOKBACK_STEPS, len(group), stride):
            seq = x_scaled[i - LOOKBACK_STEPS:i:SUBSAMPLE]
            if seq.shape != (SEQ_LEN, len(SHARED_FEATURES)) or np.isnan(seq).any():
                continue
            xs.append(seq)
            ys.append(labels[i])

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def class_weights(y):
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Training labels need both classes. Got positives={n_pos:.0f}, negatives={n_neg:.0f}")
    return {0: 1.0, 1: max(1.0, n_neg / n_pos)}


def best_f1_threshold(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    denom = precision + recall
    f1s = np.where(denom == 0, 0, 2 * precision * recall / np.maximum(denom, 1e-9))
    return float(thresholds[int(np.argmax(f1s[:-1]))]) if len(thresholds) else 0.5


def evaluate_predictions(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "n_test": int(len(y_true)),
        "n_pos_test": int((y_true == 1).sum()),
    }
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["note"] = "test split has a single class"
    return metrics, y_pred


def backup_existing_artifacts(out_dir):
    existing = [out_dir / name for name in ARTIFACTS if (out_dir / name).exists()]
    if not existing:
        return None

    backup_dir = out_dir / f"backup_bilstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def save_artifacts(model, scaler, metadata, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "bilstm_3d.weights.h5"
    keras_path = out_dir / "bilstm_3d_synthetic.keras"
    scaler_path = out_dir / "transfer_scaler.pkl"
    metadata_path = out_dir / "bilstm_3d_metadata.json"

    model.save_weights(str(weights_path))
    model.save(str(keras_path))
    joblib.dump(scaler, scaler_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        "weights": str(weights_path),
        "keras_model": str(keras_path),
        "scaler": str(scaler_path),
        "metadata": str(metadata_path),
    }


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    input_csv = Path(args.input_csv).resolve()
    out_dir = Path(args.out_dir).resolve()

    print("\n" + "=" * 72)
    print("HGW Bi-LSTM 3-day training")
    print("=" * 72)
    print(f"Input CSV : {input_csv}")
    print(f"Output dir: {out_dir}")
    print(f"Config    : lookback={LOOKBACK_HOURS}h, horizon={HORIZON_HOURS}h, seq={SEQ_LEN}x{len(SHARED_FEATURES)}")

    df, source_mode = load_training_frame(
        input_csv,
        force_rebuild_label=args.force_rebuild_label,
        keep_current_incidents=args.keep_current_incidents,
    )
    print(f"\nPrepared rows: {len(df):,} ({source_mode})")
    print(f"Period       : {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Positives    : {int(df[LABEL_COL].sum()):,} ({df[LABEL_COL].mean() * 100:.2f}%)")
    if df.empty:
        raise ValueError(
            "No usable rows after preprocessing. For a short raw CSV, try --keep-current-incidents "
            "or use a longer dataset."
        )

    df_train, df_val, df_test = temporal_split(df, args.train_ratio, args.val_ratio)
    print(f"Split rows   : train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")

    scaler = StandardScaler()
    scaler.fit(df_train[SHARED_FEATURES].values)

    x_train, y_train = build_sequences(df_train, scaler, stride=args.stride)
    x_val, y_val = build_sequences(df_val, scaler, stride=args.stride)
    x_test, y_test = build_sequences(df_test, scaler, stride=args.stride)
    print(f"Sequences    : train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")
    print(f"Seq positives: train={int(y_train.sum())}, val={int(y_val.sum())}, test={int(y_test.sum())}")

    if args.dry_run:
        print("\nDry run OK: data and sequences are ready. No model was trained.")
        return

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Not enough sequences. Need at least 24h lookback inside train/val/test splits.")

    weights = class_weights(y_train)
    print(f"Class weights: {weights}")

    backup_dir = None
    if not args.no_backup:
        backup_dir = backup_existing_artifacts(out_dir)
        if backup_dir:
            print(f"Backup       : {backup_dir}")

    model = build_bilstm()
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="prauc"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()

    tmp_dir = out_dir / f"_training_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    best_weights_path = tmp_dir / "best.weights.h5"

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="val_prauc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_prauc", mode="max", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    print("\nTraining...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )

    if best_weights_path.exists():
        model.load_weights(str(best_weights_path))

    y_prob = model.predict(x_test, batch_size=args.batch_size, verbose=0).flatten()
    threshold = best_f1_threshold(y_test, y_prob)
    metrics, y_pred = evaluate_predictions(y_test, y_prob, threshold)

    print("\n" + "=" * 72)
    print("FINAL RESULTS - Bi-LSTM 3-day")
    print("=" * 72)
    print(f"Threshold : {threshold:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"PR-AUC    : {metrics['pr_auc']:.4f}")
    print(f"Confusion : TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Normal", "Incident"], zero_division=0))

    metadata = {
        "pipeline": "Bi-LSTM 3-day horizon",
        "label_column": LABEL_COL,
        "source_mode": source_mode,
        "training_timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "Bi-LSTM kept as Tier 2 long-horizon model. "
            "Artifacts are compatible with predictor/long_horizon_dl/bilstm_loader.py."
        ),
        "config": {
            "granularity_min": GRANULARITY_MIN,
            "lookback_hours": LOOKBACK_HOURS,
            "horizon_hours": HORIZON_HOURS,
            "subsample": SUBSAMPLE,
            "seq_len": SEQ_LEN,
            "features": SHARED_FEATURES,
            "architecture": "Bi-LSTM 48->24 + Dense 32->16, no BatchNormalization",
            "loss": "binary_crossentropy with class weights",
            "optimizer": f"Adam lr={args.learning_rate}",
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "stride": args.stride,
        },
        "data": {
            "input_csv": str(input_csv),
            "prepared_rows": int(len(df)),
            "period_start": str(df["timestamp"].min()),
            "period_end": str(df["timestamp"].max()),
            "positive_rate": float(df[LABEL_COL].mean()),
            "split_rows": {
                "train": int(len(df_train)),
                "val": int(len(df_val)),
                "test": int(len(df_test)),
            },
            "sequence_shapes": {
                "train": list(x_train.shape),
                "val": list(x_val.shape),
                "test": list(x_test.shape),
            },
        },
        "metrics_synthetic_test": metrics,
        "history": {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
    }

    paths = save_artifacts(model, scaler, metadata, out_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("\nSaved artifacts:")
    for name, path in paths.items():
        print(f"  {name:<12} -> {path}")
    if backup_dir:
        print(f"\nOld artifacts kept in: {backup_dir}")


if __name__ == "__main__":
    main()
