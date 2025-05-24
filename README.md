
# EEG Eye State Prediction – Dual Approach Project

## Project Overview
This project explores **EEG-based eye state detection** using two parallel implementations. Both versions use data from a 14-channel Emotiv EPOC headset to classify whether a subject's eyes are **open or closed**, leveraging signal processing and machine learning techniques.

---

## Implementations Included

### 🔹 Extracting_code1.py
A **comprehensive and advanced pipeline** that includes:
- Signal filtering, wavelet feature extraction, and Hjorth parameters.
- Interactive time-frequency visualizations and topographic maps.
- Advanced model evaluation including learning curves, ROC analysis, and connectivity network plots.
- Results saved as plots (`spectrogram_O1.png`, `learning_curve.png`, `roc_curve.html`, etc.).

### 🔹 Extracting_code2.py
A **lightweight and modular implementation** that includes:
- Cleaned ARFF file parsing using SciPy.
- Streamlined EEG dashboard showing PSD, time series, and Random Forest importances.
- Simple train/test and confusion matrix evaluation.
- Efficient fallback using cross-validation.
- Results saved as `eeg_dashboard.png`.

---

## Files Included

- `Extracting_code1.py` — Full-featured analysis and visualization pipeline.
- `Extracting_code2.py` — Concise version for modular dashboard-based analysis.
- `eeg_dashboard.png`, `ConfusionMatrix.png`, `learning_curve.png`, etc. — Visual outputs.
- `Report.docx` — Project documentation and explanation.
- `eeg_data/EEG_Eyes.arff` — EEG dataset file (automatically downloaded or added manually).

---

## How to Run

**Step 1:** Install dependencies
```bash
pip install numpy pandas matplotlib seaborn mne pywt scikit-learn plotly
```

**Step 2:** Run your chosen version
```bash
python Extracting_code1.py   # for in-depth analysis
python Extracting_code2.py   # for quick dashboard and evaluation
```

Each script will:
- Download and parse EEG data.
- Visualize EEG signals.
- Train a Random Forest classifier.
- Evaluate the model and save visualizations.

---

## Dataset Source
- **EEG Eye State Dataset**  
- **Authors:** Oliver Rösler & David Suendermann  
- [Dataset Link](https://suendermann.com/corpus/EEG_Eyes.arff.gz)

> If automatic download fails, place the file manually in the `eeg_data/` directory.

---

## Contact

For questions, contact:
- **Pujith Kotha**
- **Aniruddh Attignal**
