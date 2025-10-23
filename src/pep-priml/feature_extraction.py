import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import entropy, skew, kurtosis

from pep_pri_ml.io_utils import load_data
from pep_pri_ml.preprocessing import moving_average_smooth

def extract_transition_segment(x, y, mode, window_sizes=np.arange(30, 105, 5)):
    """Extract transition region (max, min, eq, peak, well) from a signal."""
  
    # Take the first and second derivatives of the trajectory
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)

    best_region = (None, None)
    best_score = -np.inf if mode in ["max", "peak", "well"] else np.inf

    # Extract windowed segment if mode is min, max, or eq
    if mode in ["max", "min", "eq"]:
        if mode == 'max': # increasing
            best_score = -np.inf
            grad_mode = dy
            selector = np.argmax
        elif mode == 'min': # decreasing
            best_score = np.inf
            grad_mode = dy
            selector = np.argmin
        elif mode == 'eq':  # flattest
            best_score = np.inf
            grad_mode = np.abs(dy)
            selector = np.argmin
        else:
            raise ValueError("Mode must be 'max', 'min', or 'eq'")
        
        # Check multiple window sizes to best capture the segment
        for w in window_sizes:
            kernel = np.ones(w) / w
            avg_grad = np.convolve(grad_mode, kernel, mode='valid')
            i = selector(avg_grad)

            # Calculate a 'score'. Encourage longer windows slightly if mean_val is similar
            mean_val = avg_grad[i]
            score = mean_val + (0.1 * mean_val * (w / max(window_sizes))) if mode == 'max' else \
                    mean_val - (0.1 * mean_val * (w / max(window_sizes)))

            if mode == 'max':
                better = score > best_score
            else:
                better = score < best_score

            # Tie-breaker prefers a longer window if scores are close
            if better or (np.isclose(score, best_score, rtol=1e-5) and (best_region[1] is None or w > best_region[1])):
                best_score = score
                best_region = (i, w)
        
        i, best_w = best_region
        return x[i:i+best_w], y[i:i+best_w]

    # Peak and well detection
    elif mode in ["peak", "well"]:
        y_in = -y if mode == "well" else y

        # Adaptive prominence search
        found = False
        prominence_base = np.std(y_in) * 2
        peaks, props = np.array([]), {}
        
        for scale in [1.0, 0.5, 0.25, 0.1]:
            peaks, props = find_peaks(y_in, prominence=prominence_base * scale, width=3)
            if len(peaks) > 0:
                found = True
                break
        
        # Fallback if no peaks are found. Resort to extracting a window around the min/max second derivative 
        if not found:
            if mode == "peak":
                peak_i = np.argmax(-d2y)  # most negative curvature
            else: # mode == 'well'
                peak_i = np.argmax(d2y)   # most positive curvature
            window = 60
            start = max(0, peak_i - window//2)
            end = min(len(y), peak_i + window//2)
            return x[start:end], y[start:end]

        # Rank detected peaks, prioritizing sharpness and prominence
        sharpness = np.abs(d2y[peaks])
        prominence = props["prominences"]
        score = sharpness * prominence

        # Find the best peak
        best_idx = np.argmax(score)
        peak_i = peaks[best_idx]

        # Dynamically extract a window around the best peak 
        best_w = int(props["widths"][best_idx]) if "widths" in props else 60
        best_w = np.clip(best_w * 2, 40, 100)

        # Ensure slicing doesn't go outside of bounds 
        start = max(0, peak_i - best_w//2) 
        end = min(len(y), peak_i + best_w//2) 

        return x[start:end], y[start:end]
    pass

def extract_fft_features(y, x, mode):
    """Extract FFT-based features for a selected signal segment."""
  
    # Extract transitions segment given the mode
    x_segment, y_segment = extract_transition_segment(x, y, mode)

    y = np.asarray(y_segment)
    x = np.asarray(x_segment)
    
    # Detrend y
    y_detrended = y - np.mean(y)
    windowed = y_detrended * np.hanning(len(y))

    N = len(y)
    dt = np.mean(np.diff(x)) 
    yf = fft(windowed)
    xf = fftfreq(N, dt)[:N // 2]
    power = (2.0 / N) * np.abs(yf[:N // 2]) 

    # Calculate spectral entropy
    power_norm = power / np.sum(power) if np.sum(power) > 0 else np.ones_like(power) / len(power)
    spec_entropy = entropy(power_norm)

    # Calculate the frequency and power of the largest frequency component 
    peak_idx = np.argmax(power)
    peak_freq = xf[peak_idx]
    peak_power = power[peak_idx]

    return peak_freq, peak_power, spec_entropy
    pass

def extract_features(protein, cv_data, time_data, log_data):
    """Combine all feature extraction steps for one protein trial."""
  
    # Integration bounds
    match = log_data[log_data["protein"] == protein]
    lower, upper = float(match["lower"].values[0]), float(match["upper"].values[0])

    # Bounded data
    time_b = time_data[(time_data['r'] >= lower) & (time_data['r'] <= upper)]
    cv_b = cv_data[(cv_data['r_bin'] >= lower) & (cv_data['r_bin'] <= upper)]

    # Columns to smooth
    cols = ['RMSD', 'E_metad_r', 'r', 'Rg', 'SASA', 'n_hbond', 'n_saltbr', 'n_cont_CA', 'n_cont_heavy']
    smooth = {col: moving_average_smooth(time_b[col]) for col in cols}
    step = time_b['step']

    # AUCs
    aucs = {
        'force_auc': np.trapz(y=cv_b['force'], x=cv_b['r_bin']),
        'pmf_auc': np.trapz(y=cv_b['pmf'], x=cv_b['r_bin']),
        **{f"{col.lower()}_auc": np.trapz(y=time_b[col], x=step) for col in ['E_metad_r', 'n_cont_CA', 'n_cont_heavy', 'n_hbond', 'n_saltbr', 'SASA', 'r', 'RMSD', 'Rg']}
    }

    # Max/min
    maxs = {f"{col.lower()}_max": time_b[col].max() for col in ['E_metad_r', 'n_hbond', 'n_saltbr', 'n_cont_CA', 'n_cont_heavy', 'RMSD', 'Rg']}
    maxs['force_max'] = cv_b['force'].max()
    mins = {'sasa_min': time_b['SASA'].min()}

    # Means, stds, skew, kurtosis
    stds = {f"std_{col}": np.std(time_b[col]) for col in ['n_hbond', 'n_saltbr']}
    skews = {'skew_r': skew(time_b['r']), 'skew_emeta': skew(time_b['E_metad_r'])}
    kurts = {'kurt_r': kurtosis(time_b['r']), 'kurt_emeta': kurtosis(time_b['E_metad_r'])}

    # Time-based
    time = time_b['step'].max()

    # Derivative features
    sasa_delta = time_b['SASA'].iloc[-1] - time_b['SASA'].iloc[0]
    dr_dt_max = np.gradient(smooth['r'], step).max()
    dr2_dt2_max = np.gradient(np.gradient(smooth['r'], step), step).max()

    # fft_features
    fft_modes = ['max', 'min', 'eq', 'peak', 'well']
    fft_features_extract = ['RMSD', 'Rg']
    fft_features = {}
    for col in fft_features_extract:
        for mode in fft_modes:
            pf, pp, se = extract_fft_features(smooth[col], step, mode)
            prefix = f"{mode}_{col.lower()}"
            fft_features[f"{prefix}_peak_freq"] = pf
            fft_features[f"{prefix}_peak_power"] = pp
            fft_features[f"{prefix}_spec_entropy"] = se

    # Combine all features
    features = {
        **aucs, **maxs, **mins, **stds, **skews, **kurts,
        'time': time, 
        'sasa_delta': sasa_delta, 
        'dr_dt_max': dr_dt_max,
        'dr2_dt2_max': dr2_dt2_max,
        **fft_features
    }

    return features
    pass

def extract_all_features(config):
    """Top-level function to iterate over proteins and save combined features."""
  
    # Collect features and standard deviation over all trials
    all_features = []
    all_std = []
    
    for protein in protein_folders:
        folder_path = os.path.join(root_dir, protein)
        trial_features = []
    
        for trial in range(1, 11):
            # Find files for this trial
            cv_file = [f for f in os.listdir(folder_path) if f.endswith(f"{trial}.dat") and f.startswith("feat_cv")]
            time_file = [f for f in os.listdir(folder_path) if f.endswith(f"{trial}.dat") and f.startswith("feat_time")]
    
            # Sanity check: exactly one file each
            if len(cv_file) != 1 or len(time_file) != 1:
                print(f"Missing or too many files for {protein} trial {trial}")
                continue
    
            cv_path = os.path.join(folder_path, cv_file[0])
            time_path = os.path.join(folder_path, time_file[0])
    
            cv_data = load_data(cv_path)
            time_data = load_data(time_path)
    
            # Extract features
            features = extract_features(protein, cv_data, time_data, log_data) 
            trial_features.append(features)
    
        df = pd.DataFrame(trial_features)  # shape - (10 trials, n_features)
        
        # Average trails together
        mean_features = df.mean()
        all_features.append(mean_features)
        
        # Find standard deviation of trials
        std_features = df.std()
        all_std.append(std_features)
    
    # Final DataFrame
    std_df = pd.DataFrame(all_std)
    features_df = pd.DataFrame(all_features)

    return (features_df, std_df
    pass
