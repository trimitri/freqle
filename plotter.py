"""Plotting some meaning out of frequency series."""

from itertools import cycle
from math import floor, log10
from typing import Any, Dict, List, Tuple, Union

from ballpark import business as ballpark  # human-readable numbers
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

from .freq_series import FreqSeries
from .fbg_util.decorators import static_variable
from . import statistics as stat

_VERBOSE_METHOD_NAMES = {'adev': "Allan Deviation",
                         'oadev': "Overlapping Allan Deviation",
                         'totdev': "Total Deviation [Howe 2000]"}
_FIG_WIDTH = 8.  # Figure width in inches.
_DEFAULT_ASPECT_RATIO = 3/2
_ERR_ALPHA = .3  # Opacity of the shaded "error" regions.


def plot_asds(densities: List[stat.Asd],
              aspect: float = _DEFAULT_ASPECT_RATIO,
              figure: matplotlib.figure.Figure = None,
              merge_labels: bool = False,
              plot_options: Dict[str, Any] = None) -> matplotlib.figure.Figure:
    """Plot an ASD."""
    plot_options = {} if plot_options is None else plot_options
    fig = create_figure(aspect=aspect) if figure is None else figure
    asds = densities if isinstance(densities, list) else [densities]
    for asd in asds:
        style = _generate_line_props(asd.measurement)
        plt.loglog(asd.freqs, asd.ampls,
                   label=_label(asd.measurement, merge_labels),
                   **style, **plot_options)
        if asd.errors is not None:
            plt.gca().fill_between(asd.errors[0], asd.errors[1], asd.errors[2],
                                   color=style['color'], alpha=.3)
    plt.xlabel("Frequency in Hz")
    plt.ylabel(r"Frequency noise ASD in Hz/$\sqrt{\mathrm{Hz}}$")
    _loglog_grid()
    plt.legend()
    return fig


def save(figure: matplotlib.figure.Figure, file_name: str) -> None:
    """Save the figure for publication.

    This applies some default export settings.
    """
    figure.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=600)
    # NOTE that `pad_inches=0` can cause crammed layout in subplot arrangements.


def plot_deviations(deviations: List[stat.Adev],
                    figure: matplotlib.figure.Figure = None,
                    merge_labels: bool = False,
                    aspect: float = _DEFAULT_ASPECT_RATIO) -> matplotlib.figure.Figure:
    """An improved Allan deviation working with circular data sets.

    :param show_error: Plot error "bars".
    """
    fig = create_figure(aspect=aspect) if figure is None else figure
    devs = [deviations] if isinstance(deviations, stat.Adev) else deviations
    for dev in devs:
        style = _generate_line_props(dev.measurement)
        plt.loglog(dev.taus, dev.devs,
                   label=_label(dev.measurement, merge_labels), **style)
        if dev.errors is not None:
            plt.gca().fill_between(dev.errors[0], dev.errors[1], dev.errors[2],
                                   color=style['color'], alpha=_ERR_ALPHA)
        method = dev.method_name
    plt.legend()
    plt.xlabel("Averaging Time τ in s")
    plt.ylabel(_VERBOSE_METHOD_NAMES[method])
    _loglog_grid()
    return fig


def plot_freq(measurement: Union[FreqSeries, List[FreqSeries]],
              figure: matplotlib.figure.Figure = None,
              merge_labels: bool = False,
              offset: float = None,
              scatter: bool = False,
              tight: bool = True) -> Tuple[matplotlib.figure.Figure, float]:
    """Plot one or more frequency timelines.

    :param measurement: One or more (list of) FreqSeries to plot.
    :param figure: Use this figure instead of creating one.
    """
    fig = create_figure() if figure is None else figure
    mmts: List[FreqSeries] = [measurement] if isinstance(measurement, FreqSeries) else measurement

    # Calculate offset to substract for better display.
    min_value = min([min(mmt.data) for mmt in mmts])
    if not min_value > 0:
        raise ValueError("Invalid beat note value {}.".format(min_value))
    power = int(floor(log10(min_value)) - 2)
    ax_offset = int(floor(min_value / 10**power) * 10**power) if offset is None else offset

    for mmt in mmts:
        if scatter:
            plt.scatter(mmt.float_index, mmt.data.values - ax_offset,
                        label=_label(mmt, merge_labels), alpha=.4)
        else:
            plt.plot(mmt.float_index, mmt.data.values - ax_offset,
                     label=_label(mmt, merge_labels), linewidth=1)

    plt.xlabel("Time in seconds")
    plt.ylabel("Beat frequency (\N{MINUS SIGN}{}Hz)".format(_pretty(ax_offset)))

    axes = plt.gca()
    axes.yaxis.set_major_formatter(EngFormatter(unit="Hz", sep='\N{THIN SPACE}'))
    axes.autoscale(tight=tight, axis='x')
    plt.legend()
    plt.grid(which='both')
    return (fig, ax_offset)


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


@static_variable('prev_session', None)
def _label(data: FreqSeries, merge_labels: bool = False) -> str:
    if merge_labels and data.session == _label.prev_session:
        return ""
    _label.prev_session = data.session
    label = "{}".format(data.session) if data.session else ""
    if not merge_labels:
        label += ": {}Hz for {}s".format(_pretty(data.sample_rate),
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


def create_figure(aspect: float = _DEFAULT_ASPECT_RATIO) -> matplotlib.figure.Figure:
    """Just create an empty figure using default settings.

    :param aspect: Figure aspect ratio.
    """
    # Setting the figure width will set the "estimated bounding box" to that
    # size. After rigorous cropping, the resulting PDF will be smaller.
    return plt.figure(figsize=(_FIG_WIDTH, _FIG_WIDTH / aspect))
