"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    X = np.array(X, dtype=float)
    mask = np.isnan(X)
    counts = (~mask).sum(axis=0)
    sums = np.where(mask, 0.0, X).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros(X.shape[1]), where=counts > 0)
    return np.where(mask, means, X)

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    X = np.asarray(X, dtype = float)
    q1 = np.percentile(X, 25, axis = 0)
    q3 = np.percentile(X, 75, axis = 0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    X_clip = np.asarray(X, dtype = float)
    X_clip = np.clip(X_clip, lower, upper)
    return X_clip

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    numerator = np.asarray(numerator, dtype = float)
    denominator = np.asarray(denominator, dtype = float)
    return numerator/(denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X = np.asarray(X, dtype = float)
    new_X = np.column_stack((X, col))
    return new_X

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    labels = np.asarray(labels)
    classes, idx = np.unique(labels, return_inverse = True)
    return np.eye(len(classes))[idx]

# Step 7 - fit_standardizer
def fit_standardizer(X):
    X = np.asarray(X, dtype = float)
    mean = X.mean(axis = 0)
    std = X.std(axis = 0)
    std = np.where(std == 0, 1.0, std)
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    X = np.asarray(X, dtype= float)
    mean = np.asarray(mean, dtype= float)
    std = np.asarray(std, dtype= float)
    Z = (X-mean)/std
    return Z

# Step 9 - add_bias_column
def add_bias_column(X):
    X = np.asarray(X, dtype = float)
    ones = np.ones(X.shape[0], dtype = float)
    X_ones = np.column_stack((ones, X))
    return X_ones

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    return np.random.default_rng(seed = seed).permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    indices = np.asarray(indices, dtype = int)
    n = indices.shape[0]
    first = int(train_ratio * n)
    second = first + int(val_ratio * n)
    train_idx = indices[:first]
    val_idx = indices[first:second]
    test_idx = indices[second:]
    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = float)
    indices = np.asarray(indices, dtype = int)

    X_sub = np.take(X, indices, axis = 0)
    y_sub = np.take(y, indices, axis = 0)

    return X_sub, y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = float)

    XtX = X.T @ X
    Xty = X.T @ y.reshape(-1,1)

    theta = np.linalg.solve(XtX, Xty)
    return theta.flatten()

# Step 14 - ols_predict
def ols_predict(X, theta):
    X = np.asarray(X, dtype = float)
    theta = np.asarray(theta, dtype = float)
    pred = X @ theta
    return pred

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)

    diff = np.abs(y_true-y_pred)
    return diff.mean()

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)
    diff_sq = (y_true-y_pred)**2
    return np.sqrt(diff_sq.mean())

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)
    y_true_mean = y_true.mean()
    SS_res = np.sum((y_pred - y_true)**2)
    SS_tot = np.sum((y_true - y_true_mean)**2)
    R2 = 1 - SS_res/SS_tot if SS_tot != 0 else 0.0
    return R2

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)

    res = y_true - y_pred
    keys = ['mean', 'std', 'median_abs']
    vals = [res.mean(), res.std(), np.median(np.abs(res))]
    summary = dict(zip(keys, vals))
    return summary

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

