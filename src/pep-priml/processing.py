import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def moving_average_smooth(y, window_size=30):
    """Smooth series using a centered moving average."""
    return pd.Series(y).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()

def remove_low_variance(df, threshold=0.0):
    """Remove features with no variance."""
    sel = VarianceThreshold(threshold)
    sel.fit(df)
    return df[df.columns[sel.get_support()]]

def remove_high_correlation(df, corr_thresh=0.8):
    """Collapse highly correlated features."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
    correlated = [row for row in upper.index if upper.loc[row, col] > corr_thresh]
    if correlated:
        group = [col] + correlated
        # pick feature with larger variance
        variances = clean_features[group].var()
        keep = variances.idxmax()
        drop = [f for f in group if f != keep]
        to_drop.update(drop)
    return df.drop(columns=to_drop, errors='ignore')
