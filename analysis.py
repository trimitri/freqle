"""Getting some meaning out of frequency series."""

from typing import List

from ballpark import ballpark  # human-readable numbers
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal

from .model.freq_series import FreqSeries

def amplitude_spectral_density(data: List[FreqSeries],
                               welch_args: dict = {}) -> matplotlib.figure.Figure:
    """Plot an ASD estimated by Welch's method.

    :param welch_args: Arguments passed to `scipy.signal.welch`.
    """
    fig = plt.figure()
    for measurement in data:
        asd = signal.welch(measurement.data, measurement.sample_rate, **welch_args)
        plt.loglog(asd[0][1:], asd[1][1:],  # Don't include the zeros that welch() returns.
                   label="{}Hz for {}s".format(
                       ballpark(measurement.sample_rate),
                       ballpark(measurement.duration)))
    plt.legend()
    return fig
