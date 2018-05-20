"""Getting some meaning out of frequency series."""

from typing import List

import allantools
from ballpark import ballpark  # human-readable numbers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .model.freq_series import FreqSeries


def amplitude_spectral_density(measurements: List[FreqSeries],
                               welch_args: dict = {}) -> matplotlib.figure.Figure:
    """Plot an ASD estimated by Welch's method.

    :param welch_args: Arguments passed to `scipy.signal.welch`.
    """
    fig = plt.figure()
    for mmt in measurements:
        psd = signal.welch(mmt.data, mmt.sample_rate, **welch_args)
        plt.loglog(psd[0][1:], np.sqrt(psd[1][1:]), label=_label(mmt))
        # Don't include the zeros that welch() returns.
    plt.legend()
    return fig


def total_deviation(measurements: List[FreqSeries],
                    n_taus: int = 200) -> matplotlib.figure.Figure:
    """An improved Allan deviation working with circular data sets.

    :param n_taus: Number of different τ's to use during calculation of each
                    different measurement. (Plot accuracy)
    """
    plt.figure()
    for mmt in measurements:
        taus = np.geomspace(4/mmt.sample_rate, mmt.duration/4, num=n_taus)
        # Conservative estimate for meaningful τ values based on data.

        tau, adev, error, _ = allantools.totdev(mmt.data, rate=mmt.sample_rate,
                                                taus=taus)
        if mmt.org_freq:
            # Scale to original oscillator frequency
            adev /= mmt.org_freq
            error /= mmt.org_freq
        plt.loglog(tau, adev, label=_label(mmt))
        plt.legend()
        plt.gca().fill_between(tau, adev - error, adev + error, alpha=.4)


def _label(data: FreqSeries) -> str:
    label = "{}: ".format(data.session) if data.session else ""
    label += "{}Hz for {}s".format(ballpark(data.sample_rate),
                                   ballpark(data.duration))
    return label
