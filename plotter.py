"""Plotting some meaning out of frequency series."""

from itertools import cycle
from typing import Callable, Dict, List, NamedTuple

import allantools
from ballpark import ballpark  # human-readable numbers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .freq_series import FreqSeries
from .fbg_util.decorators import static_variable


class Adev(NamedTuple):
    """A calculated allan (or similar) deviation; ready to plot."""
    taus: np.ndarray
    devs: np.ndarray
    error: np.ndarray = None


def amplitude_spectral_density(measurements: List[FreqSeries],
                               welch_args: dict = {}) -> matplotlib.figure.Figure:
    """Plot an ASD estimated by Welch's method.

    :param welch_args: Arguments passed to `scipy.signal.welch`.
    """
    fig = _create_figure()
    for mmt in measurements:
        psd = signal.welch(mmt.data, mmt.sample_rate, **welch_args)
        plt.loglog(psd[0][1:], np.sqrt(psd[1][1:]), label=_label(mmt),
                   **_generate_line_props(mmt))
        # Don't include the zeros that welch() returns.
    plt.xlabel("Frequency in Hz")
    plt.ylabel(r"Amplitude Spectral Density in Hz/$\sqrt{\mathrm{Hz}}$")
    _loglog_grid()
    plt.legend()
    return fig


def save(figure: matplotlib.figure.Figure, file_name: str) -> None:
    """Save the figure for publication.

    This applies some default export settings.
    """
    figure.savefig(file_name, bbox_inches='tight', pad_inches=0, format='pdf')


def total_deviation(
        measurements: List[FreqSeries],
        n_taus: int = 200,
        allowable_irregularity: float = 1.05) -> matplotlib.figure.Figure:
    """An improved Allan deviation working with circular data sets.

    :param n_taus: Number of different τ's to use during calculation of each
                different measurement. (Plot accuracy)
    :param show_error: Plot error "bars".
    :param allowable_irregularity: Deviations in sample rate deemed acceptable.
                See `FreqSeries`'s documentation on the `sampling_regularity`
                property.
    :raises ValueError: The data was sampled at a rate too uneven. If this is
                known and allowable, consider setting `allowable_irregularity`.
    """
    fig = _create_figure()
    for mmt in measurements:
        dev = _calculate_deviation(mmt, n_taus, allowable_irregularity)
        plt.loglog(dev.taus, dev.devs, label=_label(mmt), **_generate_line_props(mmt))
        if dev.error:
            plt.gca().fill_between(dev.taus, dev.devs - dev.error,
                                   dev.devs + dev.error, alpha=.4)
    plt.legend()
    plt.xlabel("Averaging Time τ in s")
    plt.ylabel("Total Deviation [Howe 2000] in Hz/Hz")
    _loglog_grid()
    return fig


def _calculate_deviation(measurement: FreqSeries, n_taus: int,
                         allowable_irregularity: float,
                         algorithm: Callable = allantools.totdev) -> Adev:
    """Calculate an allan-like deviation for given data."""
    mmt = measurement
    if mmt.sampling_regularity > allowable_irregularity:
        raise ValueError("Series is too irregular in sample rate.")

    taus = np.geomspace(4/mmt.sample_rate, mmt.duration/4, num=n_taus)
    # Conservative estimate for meaningful τ values based on data.

    tau, adev, error, _ = algorithm(mmt.data.data, data_type='freq',
                                    rate=mmt.sample_rate, taus=taus)
    if mmt.org_freq:
        # Scale to original oscillator frequency
        adev /= mmt.org_freq
        error /= mmt.org_freq
    return Adev(tau, adev, None)


@static_variable('prev_style', None)
def _generate_line_props(mmt: FreqSeries) -> Dict:
    """Cycle colors and line styles according to measurement sessions.

    The `_generate_line_props.prev_style` static variable will be used to
    retain the last used style between calls.
    """
    if (_generate_line_props.prev_style is None
            or _generate_line_props.prev_style['session'] != mmt.session):
        # Start new line style cycle and new color.
        _generate_line_props.prev_style = {
            'color': next(plt.gca()._get_lines.prop_cycler)['color'],
            'session': mmt.session,
            'style': _get_style_cycler()}

    return {'linestyle': next(_generate_line_props.prev_style['style']),
            'color': _generate_line_props.prev_style['color']}


def _get_style_cycler() -> cycle:
    styles = ['-', '-.', '--', (0, [5, 1, 1, 1, 1, 1])]  # Last: ..-..-
    return cycle(styles)


def _label(data: FreqSeries) -> str:
    label = "{}: ".format(data.session) if data.session else ""
    label += "{}Hz for {}s".format(_pretty(data.sample_rate),
                                   _pretty(data.duration))
    return label


def _loglog_grid() -> None:
    """Plot a grid into the current loglog plot."""
    plt.grid(which='both')


def _pretty(number: float) -> str:
    actual_SI = {  # \u2009 is a thin space.
        24: '\u2009Y', 21: '\u2009Z',
        18: '\u2009E', 15: '\u2009P', 12: '\u2009T',
        9: '\u2009G', 6: '\u2009M', 3: '\u2009k',
        0: '\u2009',
        -3: '\u2009m', -6: '\u2009µ', -9: '\u2009n',
        -12: '\u2009p', -15: '\u2009f', -18: '\u2009a',
        -21: '\u2009z', -24: '\u2009y'}
    return ballpark(number, prefixes=actual_SI)


def _create_figure() -> matplotlib.figure.Figure:
    # Setting the figure width will set the "estimated bounding box" to that
    # size. After rigorous cropping, the resulting PDF will be smaller.
    width = 8
    height = 2/3 * width
    return plt.figure(figsize=(width, height))
