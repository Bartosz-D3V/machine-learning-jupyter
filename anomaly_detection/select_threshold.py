import numpy as np


def select_threshold(y_val, p_val):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    min_p_val = min(p_val)
    max_p_val = max(p_val)
    step_size = (max_p_val - min_p_val) / 1000
    for epsilon in np.arange(min_p_val, max_p_val, step_size):
        predicted_outcome = (p_val < epsilon)[:, np.newaxis]
        tp = np.sum(predicted_outcome[y_val == 1] == 1)
        fp = np.sum(predicted_outcome[y_val == 1] == 0)
        fn = np.sum(predicted_outcome[y_val == 0] == 1)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = (2 * prec * rec) / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon
