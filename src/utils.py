import numpy as np


def spike_detect(x, t, thresh):
    """
    Spike detection via peak finding.
    :param x: array of values
    :param t: array of times
    :param thresh: threshold for x above which is considered part of a peak
    :return: spike times
    """

    # time indexes above the threshold
    above_idxs = np.where(x > thresh * x.max())[0]

    # each run of consecutive indexes is considered one peak
    peaks = np.split(above_idxs, np.where(np.diff(above_idxs) != 1)[0] + 1)

    # the maximum signal at each peak is considered the time of that peak
    if len(peaks) > 1:
        spikes = [t[peak[0] + np.argmax(x[peak])] for peak in peaks]
    else:
        spikes = []

    return spikes


def make_edges(x, log=False):
    if log:
        x = np.log(x)
    dx = x[1] - x[0]
    x_edges = np.r_[x, x[-1] + dx] - 0.5 * dx
    if log:
        x_edges = np.exp(x_edges)
    return x_edges


