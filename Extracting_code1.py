import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import requests
import gzip
import io
import os

plt.style.use('seaborn-v0_8-whitegrid')
colors = sns.color_palette("viridis", 8)

def download_eeg_dataset(url, save_path):
    """
    Download and extract the EEG Eye State dataset with improved error handling.
    """
    print(f"Downloading EEG dataset from {url}...")
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download dataset, status code: {response.status_code}")
        
        try:
            with gzip.open(io.BytesIO(response.content), 'rt') as f:
                content = f.read()
                
            with open(save_path, 'w') as f:
                f.write(content)
                
            print(f"Dataset successfully downloaded and saved to {save_path}")
            return save_path
        
        except Exception as e:
            print(f"Error extracting gzipped content: {e}")
            try:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"Raw content saved to {save_path}")
                return save_path
            except Exception as e2:
                print(f"Error saving raw content: {e2}")
                return None
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def parse_arff_file(file_path):
    """
    Parse an ARFF file into a pandas DataFrame with improved robustness.
    """
    try:
        print(f"Parsing ARFF file: {file_path}")
        headers = []
        data_lines = []
        data_section = False
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith('%'):
                    continue
                
                if line.lower().startswith('@attribute'):
                    parts = line.split()
                    attr_name = parts[1].strip("'")
                    headers.append(attr_name)
                
                elif line.lower().startswith('@data'):
                    data_section = True
                
                elif data_section:
                    data_lines.append(line.split(','))
        
        df = pd.DataFrame(data_lines, columns=headers)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.fillna(df.mean(), inplace=True)
        
        print(f"ARFF file parsed successfully. Shape: {df.shape}")
        return df
    
    except Exception as e:
        print(f"Error parsing ARFF file: {e}")
        return None


def create_synthetic_eeg_data(n_samples=14980, n_channels=14):
    """
    Create more realistic synthetic EEG data with clear patterns between eye states.
    """
    print("Creating synthetic EEG data...")
    
    electrode_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                       'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    time = np.arange(n_samples) / 128
    
    data = np.zeros((n_samples, n_channels))
    
    for i in range(n_channels):
        white_noise = np.random.normal(0, 1, n_samples)
        b, a = signal.butter(2, 0.9, 'lowpass')
        pink_noise = signal.filtfilt(b, a, white_noise)
        data[:, i] = pink_noise * 2
    
    block_size = 256
    n_blocks = n_samples // block_size + 1
    
    eye_state = np.zeros(n_samples)
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, n_samples)
        eye_state[start_idx:end_idx] = i % 2
    
    alpha_freq = 10
    alpha_electrodes = {'O1': 5.0, 'O2': 5.0, 'P7': 3.0, 'P8': 3.0, 'T7': 1.0, 'T8': 1.0}
    
    for i in range(n_samples):
        if eye_state[i] == 0:
            alpha_wave = 5 * np.sin(2 * np.pi * alpha_freq * time[i])
            for e_name, scale in alpha_electrodes.items():
                e_idx = electrode_names.index(e_name)
                data[i, e_idx] += alpha_wave * scale
    
    beta_freq = 20
    beta_electrodes = {'AF3': 2.0, 'AF4': 2.0, 'F3': 1.5, 'F4': 1.5, 'F7': 1.0, 'F8': 1.0}
    
    for i in range(n_samples):
        if eye_state[i] == 1:
            beta_wave = 2 * np.sin(2 * np.pi * beta_freq * time[i])
            for e_name, scale in beta_electrodes.items():
                e_idx = electrode_names.index(e_name)
                data[i, e_idx] += beta_wave * scale
    
    artifact_times = np.random.choice(n_samples, 20, replace=False)
    for t in artifact_times:
        for i in range(n_channels):
            artifact_length = np.random.randint(10, 50)
            end_t = min(t + artifact_length, n_samples)
            data[t:end_t, i] += np.random.normal(0, 15, end_t - t)
    
    df = pd.DataFrame(data, columns=electrode_names)
    df['eyeDetection'] = eye_state
    
    print(f"Synthetic data created with shape: {df.shape}")
    return df

def plot_improved_spectrogram(signal_data, channel_idx, channel_name, fs=128, 
                             eye_states=None, figsize=(12, 8), cmap='viridis'):
    """
    Plot an improved spectrogram visualization for EEG data.
    """
    channel_data = signal_data[:, channel_idx]
    plt.figure(figsize=figsize)
    
    f, t, Sxx = signal.spectrogram(
        channel_data, 
        fs=fs, 
        nperseg=256,
        noverlap=192,
        scaling='density'
    )
    
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap=cmap)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Power/Frequency [dB/Hz]')
    plt.title(f'Spectrogram of {channel_name}')
    plt.ylim(0, 50)
    
    if eye_states is not None:
        ax = plt.gca()
        prev_state = eye_states[0]
        ax2 = ax.twinx()
        ax2.set_ylabel('Eye State')
        ax2.set_ylim(0, 1)
        
        time_points = np.arange(len(eye_states)) / fs
        ax2.step(time_points, eye_states, where='post', color='red', alpha=0.5, linewidth=1.5)
        
        for i in range(1, len(eye_states)):
            if eye_states[i] != eye_states[i-1]:
                ax.axvline(x=i/fs, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', alpha=0.5, label='Eye State (0=Closed, 1=Open)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return plt.gcf()


def plot_improved_learning_curve(estimator, X, y, title="Learning Curve", ylim=None, cv=5,
                              n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate an improved learning curve plot with better visual distinction.
    """
    plt.figure(figsize=(12, 7))
    plt.title(title, fontsize=14)
    
    if ylim is not None:
        plt.ylim(*ylim)
    else:
        plt.ylim(0.5, 1.01)
        
    plt.xlabel("Training examples", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 10),
        return_times=True
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="#FF9999")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="#FF0000", 
             label="Training score", linewidth=2, markersize=8)
    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="#99FF99")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="#009900", 
             label="Cross-validation score", linewidth=2, markersize=8)
    
    ax2 = plt.gca().twinx()
    ax2.plot(train_sizes, fit_times_mean, 'o-', color="#0000FF", 
             label="Fit time", linewidth=2, markersize=6)
    ax2.set_ylabel("Fit time (s)", color="#0000FF", fontsize=12)
    ax2.tick_params(axis='y', colors="#0000FF")
    
    plt.annotate(f"Max CV Score: {max(test_scores_mean):.4f}", 
                 xy=(0.02, 0.07), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    handles1, labels1 = plt.gca().get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="best", frameon=True,
               facecolor='white', edgecolor='gray', framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    return plt.gcf()


def plot_eeg_time_series_by_state(X, y, electrode_names, figsize=(14, 10), n_samples=1000):
    """
    Plot EEG time series data grouped by eye state for better pattern visualization.
    """
    closed_idx = np.where(y == 0)[0][:n_samples//2]
    open_idx = np.where(y == 1)[0][:n_samples//2]
    
    plt.figure(figsize=figsize)
    
    key_electrodes = ['O1', 'O2', 'P7', 'P8', 'F3', 'F4']
    electrode_indices = [electrode_names.index(e) for e in key_electrodes if e in electrode_names]
    
    for i, electrode_idx in enumerate(electrode_indices):
        plt.subplot(len(electrode_indices), 1, i+1)
        
        closed_data = X[closed_idx[:500], electrode_idx]
        time_closed = np.arange(len(closed_data)) / 128
        plt.plot(time_closed, closed_data, 'b-', alpha=0.7, label='Eyes Closed')
        
        open_data = X[open_idx[:500], electrode_idx]
        time_open = np.arange(len(open_data)) / 128
        plt.plot(time_open + time_closed[-1] + 0.5, open_data, 'r-', alpha=0.7, label='Eyes Open')
        
        plt.axvline(x=time_closed[-1] + 0.25, color='k', linestyle='--', alpha=0.5)
        
        plt.title(f'Electrode: {electrode_names[electrode_idx]}')
        plt.ylabel('Amplitude (µV)')
        
        if i == 0:
            plt.legend()
        
        if i == len(electrode_indices) - 1:
            plt.xlabel('Time (seconds)')
    
    plt.tight_layout()
    return plt.gcf()


def plot_frequency_analysis_by_electrode(X, y, electrode_names, fs=128, figsize=(14, 10)):
    """
    Plot frequency power spectrum for each electrode comparing eye states.
    """
    closed_idx = np.where(y == 0)[0]
    open_idx = np.where(y == 1)[0]
    plt.figure(figsize=figsize)
    
    key_electrodes = ['O1', 'O2', 'P7', 'P8', 'F3', 'F4', 'AF3', 'AF4']
    electrode_indices = [electrode_names.index(e) for e in key_electrodes if e in electrode_names]
    
    n_cols = 2
    n_rows = int(np.ceil(len(electrode_indices) / n_cols))
    
    for i, electrode_idx in enumerate(electrode_indices):
        plt.subplot(n_rows, n_cols, i+1)
        
        closed_data = X[closed_idx, electrode_idx]
        f_closed, psd_closed = signal.welch(closed_data, fs=fs, nperseg=256)
        
        open_data = X[open_idx, electrode_idx]
        f_open, psd_open = signal.welch(open_data, fs=fs, nperseg=256)
        
        plt.semilogy(f_closed, psd_closed, 'b-', alpha=0.7, label='Eyes Closed')
        plt.semilogy(f_open, psd_open, 'r-', alpha=0.7, label='Eyes Open')
        alpha_band = (8, 13)
        beta_band = (13, 30)
        
        plt.axvspan(alpha_band[0], alpha_band[1], color='lightblue', alpha=0.3, label='Alpha Band')
        plt.axvspan(beta_band[0], beta_band[1], color='lightgreen', alpha=0.3, label='Beta Band')
        
        plt.title(f'Electrode: {electrode_names[electrode_idx]}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (µV²/Hz)')
        plt.xlim(0, 50)      
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    return plt.gcf()


def plot_feature_importance(model, X, feature_names, figsize=(12, 6)):
    """
    Plot feature importance from the trained model with improved visualization.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(importances)))
    plt.figure(figsize=figsize)
    
    plt.bar(range(len(importances)), importances[indices], align='center', 
            color=colors[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Electrode Importance for Eye State Prediction', fontsize=14)
    plt.xlabel('Electrodes', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(importances[indices]):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', 
                 rotation=45, fontsize=8)
    
    plt.tight_layout()
    return plt.gcf()


def main():
    """
    Main function with improved visualizations and analysis.
    """
    print("EEG Eye State Analysis Implementation")
    print("=====================================")
    url = "https://suendermann.com/corpus/EEG_Eyes.arff.gz"
    data_path = "/eeg_data/EEG_Eyes.arff"
    
    downloaded_path = download_eeg_dataset(url, data_path)
    
    if downloaded_path and os.path.exists(downloaded_path):
        print("Using downloaded dataset.")
        df = parse_arff_file(downloaded_path)
        
        if df is None:
            print("Failed to parse downloaded dataset. Using synthetic data.")
            df = create_synthetic_eeg_data()
    else:
        print("Using synthetic data for visualization.")
        df = create_synthetic_eeg_data()
    
    if 'eyeDetection' in df.columns:
        y = df['eyeDetection'].values
        X = df.drop('eyeDetection', axis=1).values
        electrode_names = df.drop('eyeDetection', axis=1).columns.tolist()
    elif 'class' in df.columns:
        y = df['class'].values
        X = df.drop('class', axis=1).values
        electrode_names = df.drop('class', axis=1).columns.tolist()
    else:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
        electrode_names = df.iloc[:, :-1].columns.tolist()
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} electrodes")
    print(f"Eye state distribution: {np.unique(y, return_counts=True)}")
    print("\nCreating improved visualizations...")
    
    electrode_idx = 6
    spectrogram_fig = plot_improved_spectrogram(
        X, electrode_idx, electrode_names[electrode_idx], 
        eye_states=y, figsize=(12, 8), cmap='viridis'
    )
    spectrogram_fig.savefig("improved_spectrogram_O1.png", dpi=300, bbox_inches='tight')
    plt.close(spectrogram_fig)
    print("Improved spectrogram saved as improved_spectrogram_O1.png")
    
    time_series_fig = plot_eeg_time_series_by_state(X, y, electrode_names)
    time_series_fig.savefig("eeg_by_eye_state.png", dpi=300, bbox_inches='tight')
    plt.close(time_series_fig)
    print("Time series comparison saved as eeg_by_eye_state.png")
    
    freq_analysis_fig = plot_frequency_analysis_by_electrode(X, y, electrode_names)
    freq_analysis_fig.savefig("frequency_analysis_by_electrode.png", dpi=300, bbox_inches='tight')
    plt.close(freq_analysis_fig)
    print("Frequency analysis saved as frequency_analysis_by_electrode.png")
    print("\nTraining improved classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    learning_curve_fig = plot_improved_learning_curve(
        model, X, y, title="Random Forest Learning Curve (Tuned Hyperparameters)"
    )
    learning_curve_fig.savefig("improved_learning_curve.png", dpi=300, bbox_inches='tight')
    plt.close(learning_curve_fig)
    print("Improved learning curve saved as improved_learning_curve.png")
    
    model.fit(X, y)
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    importance_fig = plot_feature_importance(model, X, electrode_names)
    importance_fig.savefig("electrode_importance.png", dpi=300, bbox_inches='tight')
    plt.close(importance_fig)
    print("Electrode importance plot saved as electrode_importance.png")
    
    print("\nAnalysis completed successfully!")
    print("All improved visualizations have been saved to the current directory.")


if __name__ == "__main__":
    main()