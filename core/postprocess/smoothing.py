import numpy as np

def moving_average(data, window=5):
    smoothed = np.copy(data)
    for t in range(window, len(data)):
        smoothed[t] = np.mean(data[t-window:t], axis=0)
    return smoothed
