import numpy as np
from scipy import stats as sp_stats


def select_block_length(x, max_lag=40):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    var0 = np.dot(x, x) / n

    if var0 == 0:
        return 5

    bound = 1.96 / np.sqrt(n)

    for lag in range(1, min(max_lag, n // 4)):
        r = np.dot(x[:-lag], x[lag:]) / ((n - lag) * var0)
        if abs(r) < bound:
            return max(3, min(20, lag))

    return 10


print("Loaded block length selector")