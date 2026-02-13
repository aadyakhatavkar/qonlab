"""
Unified metric computation for structural break estimation.

All metrics functions take error arrays and return scalar values.
Missing/invalid outputs are represented as np.nan.
"""

import numpy as np
import scipy.stats as stats


def rmse(errors):
    """Root Mean Squared Error.
    
    Args:
        errors: 1D array of forecast errors (y_true - y_pred)
        
    Returns:
        float: RMSE value or np.nan if empty
    """
    e = np.asarray(errors, dtype=float)
    if len(e) == 0:
        return np.nan
    return float(np.sqrt(np.mean(e**2)))


def mae(errors):
    """Mean Absolute Error.
    
    Args:
        errors: 1D array of forecast errors
        
    Returns:
        float: MAE value or np.nan if empty
    """
    e = np.asarray(errors, dtype=float)
    if len(e) == 0:
        return np.nan
    return float(np.mean(np.abs(e)))


def bias(errors):
    """Mean Error (Bias).
    
    Args:
        errors: 1D array of forecast errors
        
    Returns:
        float: Bias value or np.nan if empty
    """
    e = np.asarray(errors, dtype=float)
    if len(e) == 0:
        return np.nan
    return float(np.mean(e))


def var_error(errors):
    """Variance of errors (Var(error) = E[(e - E[e])^2]).
    
    Args:
        errors: 1D array of forecast errors
        
    Returns:
        float: Variance of errors or np.nan if empty
    """
    e = np.asarray(errors, dtype=float)
    if len(e) == 0:
        return np.nan
    return float(np.var(e))


def logscore_gaussian(y_true, y_pred, sigma2):
    """Gaussian log-likelihood / log-score.
    
    Computes log p(y_true | y_pred, sigma2) under N(y_pred, sigma2) distribution.
    Negative values are worse; higher (less negative) is better.
    
    Formula: log(1/sqrt(2π*σ²)) - (y - ŷ)² / (2σ²)
             = -0.5*log(2π*σ²) - (y - ŷ)² / (2σ²)
    
    Args:
        y_true: 1D array of true values
        y_pred: 1D array of point predictions (means)
        sigma2: 1D array of predictive variances (must be > 0)
                If sigma2 contains zeros or negatives, returns np.nan
        
    Returns:
        float: Mean log-score across samples, or np.nan if invalid sigma2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    
    # Check for invalid sigma2
    if np.any(sigma2 <= 0) or np.any(np.isnan(sigma2)):
        return np.nan
    
    if len(y_true) == 0:
        return np.nan
    
    # Log likelihood for Gaussian: -0.5*log(2π*σ²) - (y-ŷ)²/(2σ²)
    log_scores = -0.5 * np.log(2 * np.pi * sigma2) - (y_true - y_pred)**2 / (2 * sigma2)
    return float(np.mean(log_scores))


def coverage_from_errors(errors, confidence=0.95):
    """Estimate coverage from error distribution.
    
    Uses error quantiles to construct prediction intervals.
    For 95% confidence, uses 2.5% and 97.5% quantiles.
    
    Args:
        errors: 1D array of forecast errors
        confidence: Confidence level (default 0.95 for 95%)
        
    Returns:
        float: Coverage as proportion (0-1)
    """
    e = np.asarray(errors, dtype=float)
    if len(e) == 0:
        return np.nan
    
    # Use quantiles of errors as PI bounds
    alpha = 1 - confidence
    q_lower = np.quantile(e, alpha/2)
    q_upper = np.quantile(e, 1 - alpha/2)
    
    # Coverage: proportion of errors within the interval
    in_interval = (e >= q_lower) & (e <= q_upper)
    return float(np.mean(in_interval))


def logscore_from_errors(errors):
    """Estimate log-score from error distribution.
    
    Uses error variance as uncertainty estimate:
    logscore ≈ -0.5*log(2π*Var(e)) - Mean(e²)/(2*Var(e))
    
    Approximation assumes 0-mean forecast errors (unbiased forecasts).
    
    Args:
        errors: 1D array of forecast errors
        
    Returns:
        float: Estimated log-score
    """
    e = np.asarray(errors, dtype=float)
    if len(e) == 0:
        return np.nan
    
    sigma2 = np.var(e)
    if sigma2 <= 0:
        return np.nan
    
    # Log-score under Gaussian: -0.5*log(2π*σ²) - e²/(2σ²)
    # Average across samples
    log_scores = -0.5 * np.log(2 * np.pi * sigma2) - (e**2) / (2 * sigma2)
    return float(np.mean(log_scores))


def coverage_95(y_true, y_pred, sigma2):
    """95% Coverage probability.
    
    Computes proportion of observations where y_true falls within the
    95% prediction interval: [ŷ - 1.96*sqrt(σ²), ŷ + 1.96*sqrt(σ²)]
    
    Args:
        y_true: 1D array of true values
        y_pred: 1D array of point predictions (means)
        sigma2: 1D array of predictive variances (must be > 0)
                If sigma2 contains zeros or negatives, returns np.nan
        
    Returns:
        float: Coverage as proportion (0-1), or np.nan if invalid sigma2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    
    # Check for invalid sigma2
    if np.any(sigma2 <= 0) or np.any(np.isnan(sigma2)):
        return np.nan
    
    if len(y_true) == 0:
        return np.nan
    
    # 95% PI: ŷ ± 1.96*sqrt(σ²)
    z_score = 1.96
    std_dev = np.sqrt(sigma2)
    lower = y_pred - z_score * std_dev
    upper = y_pred + z_score * std_dev
    
    # Count coverage
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = float(np.mean(in_interval))
    
    return coverage
