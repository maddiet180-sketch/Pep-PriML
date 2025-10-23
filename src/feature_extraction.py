root_dir = root_dir

protein_folders = ['1jp5', '1zhk', '2o9v', '2pr9', '2puy', '2qbw', '3btr', '3dri', '3i5r', '3iqq', '3jzo', '4ezq', '4gne', '4j45', '4j73', 
                   '4x3h', '5awu', '5hfb', '5mlo', '5vwi', '5xn3', '5zys', '6dql', '6nsx', '6qc0']
def load_data(file_path):
    """ 
    Returns pd.DataFrame given file path (str).
    """
    df = pd.read_csv(file_path, sep='\s+')
    return df
  
def extract_transition_segment(x, y, mode, window_sizes=np.arange(30, 105, 5)):
    """ 
    Extract a region of a trajectory signal corresponding to the mode
    
    For max, min, eq:
        * Window determined using convolution to find the region that results in the max/min 
          average slope. A scoring function is utilized to prioritize longer windows
          whenever possible.
        max - longest smoothly increasing segemnt 
        min - longest smoothly decreasing segment 
        eq - longest flat segment
    
    For peak, well:
        * Window determined using scipy.signal.find_peaks with an adaptive prominence threshold 
          based on the signals standard deviation. 
        peak - segment around the most defined peak
        well - region arounf the most defined well

    Parameters
    ----------
    x : np.ndarray
        Simulation steps (much like simulations time).
    y : np.ndarray
        Signal values (RMSD or Rg).
    mode : str
        Transition mode ('max', 'min', 'eq', 'peak', 'well').
    window_sizes: np.ndarray
        Window sizes to test (30-100 in incriments of 5).
        
    Returns
    -------
    x[window] : np.ndarray
        Selected x segement.
    y[window] : np.ndarray
        Selected y segement.
    """
  
    # Take first and second derivative of the trajectory
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
        
        # Check multiple window sizes to best capture segment
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

            # Tie-breaker prefers longer window if scores are close
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

def extract_fft_features(y, x, mode):
    """ 
    Extract FFT-based features (peak frequency, peak power, spectral entropy)
    for a selected region of a trajectory signal.

    Parameters
    ----------
    y : np.ndarray
        Signal values (RMSD or Rg).
    x : np.ndarray
        Simulation steps (much like simulation time).
    mode : str
        Transition mode ('max', 'min', 'eq', 'peak', 'well').

    Returns
    -------
    peak_freq : float
        Dominant frequency of the FFT spectrum.
    peak_power : float
        Power of the dominant frequency.
    spec_entropy : float
        Spectral entropy of the normalized power spectrum.
    """

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

def moving_average_smooth(y, window_size=30):
    """
    Smooth data using a moving average.

    Parameters
    ----------
    y : np.ndarray
        Series data.
    window_size: int
        Size of averaging window (smaller window = lower granularity).

    Returns
    -------
    y_smooth : np.ndarray
        Smoothed series data.
    """
    y_smooth = pd.Series(y).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    return y_smooth

def extract_features(protein, cv_data, time_data, log_data):
    """ 
    Extract features for one protein trial derived from cv data and time data
    within the bounds given by the log_data.

    Parameters
    ----------
    protein : str
        Name of protein.
    cv_data : pd.DataFrame
        DataFrame containing data collected as a function of the collective variable.
    time_data : pd.DataFrame
        DataFrame containing data collected as a function of steps (simulation time).
    log_data: pd.DataFrame
        DataFrame containing integration bounds for each protein.

    Returns
    -------
    features : dict
        All features extracted from one trial of this protein.
    """

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
  
# log_data gives bounds to extract features within
log_data = {
    'protein' : ['4gne', '2o9v', '4x3h', '3i5r', '3dri', '1zhk', '2puy', '4j73', '5hfb', '5awu', '5vwi', '4ezq', 
     '3btr', '5mlo', '3iqq', '4j45', '5xn3', '5zys', '6nsx', '6qc0', '2qbw', '2pr9', '6dql', '3jzo', '1jp5'],
    'lower' : [0.5, 0.5, 0.5, 0.6, 0.3, 0.5, 0.8, 0.6, 0.9, 0.6, 0.6, 0.5, 0.5, 0.6, 0.3, 0.4, 0.7, 0.8, 0.5, 0.6, 0.5, 0.6, 0.9, 0.6, 0.3],
    'upper' : [2.5, 2.5, 2.5, 2.6, 2.3, 2.5, 2.8, 2.6, 2.9, 2.6, 2.6, 2.5, 2.5, 2.6, 2.3, 2.4, 2.7, 2.8, 2.5, 2.6, 2.5, 2.6, 2.9, 2.6, 2.3],
}

log_data = pd.DataFrame.from_dict(log_data)

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

        # Extarct features
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

# Experimentally determined delta G (binding affinity values)
features_df['exp_dG'] = [-12.57, -6.07, -4.83, -10.06, -6.17, -9.89, -7.23, -6.03, -6.00, -7.55, -11.62, -7.09, 
     -3.37, -7.87, -6.21, -5.76, -6.34, -6.23, -7.25, -6.60, -8.42, -8.55, -10.75, -9.05, -9.88]
corr = features_df.corr()

# Heatmap and clustermap to visualize feature correlation
plt.figure(figsize=(12, 10))
sns.heatmap(corr, 
            annot=False, 
            cmap='viridis',
            fmt=".2f") 
plt.title('Feature Correlation')
plt.show()
sns.clustermap(corr, cmap='viridis', figsize=(12,12))

# Clean features by removing attributes with no variance or high correlation
clean_features = features_df.copy()

report = {
    'no_var': [],
    'high_corr': [],
}

# remove features with no variance 
sel = VarianceThreshold(threshold=0) 
sel.fit(clean_features)
keep_mask = sel.get_support()
keep_cols = clean_features.columns[keep_mask]
drop_cols = clean_features.columns[~keep_mask]
clean_features=clean_features[keep_cols]

report['no_var'] = drop_cols.tolist()

# Collpase highly correlated features
corr_matrix = clean_features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = set()
for col in upper.columns:
    correlated = [row for row in upper.index if upper.loc[row, col] > 0.80]
    if correlated:
        group = [col] + correlated
        # pick feature with larger variance
        variances = clean_features[group].var()
        keep = variances.idxmax()
        drop = [f for f in group if f != keep]
        to_drop.update(drop)

report['high_corr'] = list(to_drop)

clean_features = clean_features.drop(columns=to_drop)

# # Uncomment below to export the final, cleaned features as a comma-separated .csv
# clean_features.to_csv('final_features.csv')
