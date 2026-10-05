import numpy as np

def mae(pred, true):
    return np.mean(np.abs(pred - true))

def mse(pred, true):
    return np.mean((pred - true) ** 2)

def rmse(pred, true):
    return np.sqrt(mse(pred, true))

def smape(pred, true):
    """
    sMAPE = (100/n) * sum(2 * |y_pred - y_true| / (|y_true| + |y_pred|))
    """
    denominator = np.abs(true) + np.abs(pred) + 1e-8
    return np.mean(200.0 * np.abs(pred - true) / denominator)

def calculate_metrics(pred, true):
    """
    Returns all tracked metrics for the validation loop.
    """
    mae_val = mae(pred, true)
    mse_val = mse(pred, true)
    rmse_val = rmse(pred, true)
    smape_val = smape(pred, true)
    
    return mae_val, mse_val, rmse_val, smape_val
