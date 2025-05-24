import logging
import os
import gzip
import io
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import arff as arff_io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─── Constants ────────────────────────────────────────────────────────────────
FS = 128                   # sampling frequency (Hz)
NPSEG = 128                # nperseg for Welch/spectrogram
PSD_SAMPLES = 200          # points per state for PSD & time series
SPEC_SAMPLES = 300         # points per state for spectrogram
EXPECTED_CHANNELS = 14     # number of EEG channels expected
LABEL_NAME = 'eyeDetection'  # preferred label name (fallback: last column)
DATA_URL = "https://suendermann.com/corpus/EEG_Eyes.arff.gz"
DATA_PATH = "/eeg_data/EEG_Eyes.arff"

# ─── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_eeg_dataset(url: str, save_path: str) -> Optional[str]:
    """Download and, if needed, decompress the EEG ARFF file."""
    logger.info(f"Downloading EEG dataset from {url} …")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as ex:
        logger.error(f"Download failed: {ex}")
        return None

    # attempt to ungzip
    try:
        with gzip.open(io.BytesIO(resp.content), 'rt') as gz:
            content = gz.read()
        with open(save_path, 'w') as f:
            f.write(content)
        logger.info("Successfully decompressed ARFF")
    except Exception:
        # fallback: save raw bytes
        with open(save_path, 'wb') as f:
            f.write(resp.content)
        logger.warning("Saved raw content; may not be valid ARFF")

    return save_path


def parse_arff_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load an ARFF file into a DataFrame using SciPy.
    Validates channel count and decodes byte fields.
    """
    logger.info(f"Parsing ARFF file via SciPy: {file_path}")
    try:
        raw, meta = arff_io.loadarff(file_path)
    except Exception as ex:
        logger.error(f"ARFF loader error: {ex}")
        return None

    df = pd.DataFrame(raw)
    # decode bytes → str
    for col in df.select_dtypes([object]):
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # validate number of channels
    n_cols = df.shape[1]
    if n_cols not in (EXPECTED_CHANNELS + 1,):
        logger.warning(f"Expected {EXPECTED_CHANNELS+1} columns, found {n_cols}")

    # coerce non‐numeric data then fill missing
    df = df.apply(pd.to_numeric, errors='ignore')
    df.fillna(df.mean(numeric_only=True), inplace=True)

    logger.info(f"Parsed DataFrame shape: {df.shape}")
    return df


def select_electrode(electrode_names: List[str], prefs: List[str]=['O1','O2']) -> int:
    """Return the index of the first preferred electrode, else 0."""
    for p in prefs:
        if p in electrode_names:
            return electrode_names.index(p)
    return 0


def create_dashboard(X: np.ndarray, y: np.ndarray, names: List[str]) -> plt.Figure:
    """4‐panel figure: PSD, time‐series, electrode‐diff bar, RF importances."""
    logger.info("Building dashboard …")
    fig = plt.figure(figsize=(18, 14))

    closed_idx = np.where(y == 0)[0][:PSD_SAMPLES]
    open_idx   = np.where(y == 1)[0][:PSD_SAMPLES]
    e_idx = select_electrode(names)

    # ── Panel 1: PSD ─────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    f, psd_c = signal.welch(X[closed_idx, e_idx], fs=FS,   nperseg=NPSEG)
    f, psd_o = signal.welch(X[open_idx,   e_idx], fs=FS,   nperseg=NPSEG)
    ax1.semilogy(f, psd_c, label='Closed')
    ax1.semilogy(f, psd_o, label='Open')
    ax1.set(title=f"PSD – {names[e_idx]}", xlabel="Hz", ylabel="Power (dB/Hz)")
    ax1.set_xlim(0, 50)
    ax1.legend()

    # ── Panel 2: Time Series ─────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    t = np.arange(PSD_SAMPLES) / FS
    ax2.plot(t, X[closed_idx, e_idx],  label='Closed', alpha=0.7)
    ax2.plot(t + t[-1] + .2, X[open_idx, e_idx], label='Open', alpha=0.7)
    ax2.axvline(x=t[-1] + .1, linestyle='--', color='k', alpha=0.5)
    ax2.set(title=f"Time Series – {names[e_idx]}", xlabel="s", ylabel="µV")
    ax2.legend()

    # ── Panel 3: Electrode ΔBar Chart ───────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    diffs = np.abs(X[open_idx].mean(axis=0) - X[closed_idx].mean(axis=0))
    idxs = np.argsort(diffs)
    ax3.barh(range(len(names)), diffs[idxs], align='center', alpha=0.7)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(np.array(names)[idxs])
    ax3.invert_yaxis()
    ax3.set(title="Δ between Open/Closed", xlabel="|ΔµV|")
    for i in idxs[-3:]:
        ax3.get_yticklabels()[np.where(idxs == i)[0][0]].set_color('red')

    # ── Panel 4: RF Feature Importances ──────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipeline.fit(X, y)
    imps = pipeline.named_steps['clf'].feature_importances_
    order = np.argsort(imps)[::-1]
    ax4.bar(range(len(names)), imps[order], alpha=0.7)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(np.array(names)[order], rotation=45, ha='right')
    ax4.set(title="RF Feature Importances", ylabel="Importance")

    plt.tight_layout()
    return fig


def evaluate_model(X: np.ndarray, y: np.ndarray) -> None:
    """Train/test split → classification report + confusion matrix."""
    logger.info("Training/test split evaluation …")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_te)

    logger.info("\n" + classification_report(y_te, preds,
                         target_names=['Closed','Open']))
    cm = confusion_matrix(y_te, preds)
    plt.figure(figsize=(6,6))
    plt.title("Confusion Matrix")
    sns = None
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Closed','Open'],
                    yticklabels=['Closed','Open'])
    except ImportError:
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def fallback_cv(X: np.ndarray, y: np.ndarray) -> None:
    """5-fold CV accuracy if train/test or plotting errors occur."""
    logger.info("Fallback: cross-validation")
    scores = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42),
                             X, y, cv=5)
    logger.info(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")


def main():
    # ── 1) Download / Load ─────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        dl = download_eeg_dataset(DATA_URL, DATA_PATH)
        if not dl:
            logger.error("Cannot proceed without data → exiting")
            return
    df = parse_arff_file(DATA_PATH)
    if df is None:
        return

    # ── 2) Split features / target ──────────────────────────────────────
    if LABEL_NAME in df:
        label = LABEL_NAME
    else:
        label = df.columns[-1]
        logger.warning(f"Using last column '{label}' as label")

    y = df[label].values
    X = df.drop(label, axis=1).values
    names = df.drop(label, axis=1).columns.tolist()
    logger.info(f"Data: {X.shape[0]} samples × {X.shape[1]} channels")

    # ── 3) Visualize ───────────────────────────────────────────────────
    try:
        fig = create_dashboard(X, y, names)
        fig.savefig("eeg_dashboard.png", dpi=300, bbox_inches='tight')
        logger.info("Saved dashboard → eeg_dashboard.png")
    except Exception as ex:
        logger.error(f"Dashboard failed: {ex}")
        fallback_cv(X, y)

    # ── 4) Evaluate ────────────────────────────────────────────────────
    try:
        evaluate_model(X, y)
    except Exception as ex:
        logger.error(f"Evaluation failed: {ex}")
        fallback_cv(X, y)

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()