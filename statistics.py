"""Do the actual statistic analysis."""
from copy import deepcopy
from typing import Callable, List, NamedTuple
import allantools
import numpy as np

from .freq_series import FreqSeries

_DEFAULT_DEV: Callable = allantools.oadev
_OK_IRREGULARITY = 1.05

class Adev(NamedTuple):  # pylint: disable=too-few-public-methods
    """A calculated allan (or similar) deviation."""
    method_name: str
    """Method used for calculating the deviation.

    Names like in allantools:
      - adev
      - oadev
      - totdev
      - etc...
    """
    measurement: FreqSeries
    taus: np.ndarray
    devs: np.ndarray
    errors: np.ndarray = None
    """An estimate for the uncertainty of the calculated adev values.

    This array has the shape (3, n), with three rows like
      - [0] The tau value the error was estimated at
      - [1] The minimum value of the confidence interval
      - [2] The maximum value of the confidence interval

    Note that this interval must not necessary center around `.devs` exactly.
    For details see the error estimation code.
    """


def deviation(measurement: FreqSeries, estimate_error: bool = True,
              taus: int = None,
              allowable_irregularity: float = _OK_IRREGULARITY,
              method: Callable = _DEFAULT_DEV) -> Adev:
    """Calculate an allan-like deviation for given data.
    :param taus: τ's to use during calculation of each different measurement.
                (Plot accuracy)
    :param allowable_irregularity: Deviations in sample rate deemed acceptable.
                See `FreqSeries`'s documentation on the `sampling_regularity`
                property.
    :raises ValueError: The data was sampled at a rate too uneven. If this is
                known and allowable, consider setting `allowable_irregularity`.
    """
    def _estimate_error(mmt: FreqSeries, n_chops: int = 10) -> np.ndarray:
        slice_length = int(len(mmt.data) / n_chops)
        devs: List[Adev] = []
        taus: np.ndarray = None
        for idx in range(n_chops):
            chop = deepcopy(mmt)
            chop.trim(start=idx * slice_length,
                      end=(n_chops - idx - 1) * slice_length)
            if taus is None:
                taus = generate_taus(chop, until=.1)
            # Use the same set of sampling times for all chops.
            devs.append(deviation(
                chop, allowable_irregularity=allowable_irregularity,
                taus=taus, method=method, estimate_error=False))

        dev_matrix = np.array([dev.devs for dev in devs])
        dev_avg = dev_matrix.mean(0)
        dev_stdev = dev_matrix.std(0) / np.sqrt(n_chops)
        actual_tau = devs[0].taus
        return np.array([actual_tau, dev_avg - dev_stdev, dev_avg + dev_stdev])

    mmt = measurement
    if mmt.sampling_regularity > allowable_irregularity:
        raise ValueError("Series is too irregular in sample rate.")

    tau, adev, _, _ = method(mmt.data.data, data_type='freq',
                             rate=mmt.sample_rate,
                             taus=generate_taus(mmt, until=.01) if taus is None else taus)

    error = _estimate_error(mmt) if estimate_error else None

    if mmt.org_freq:
        adev /= mmt.org_freq

    return Adev(method.__name__, mmt, tau, adev, error)


def generate_taus(mmt: FreqSeries, n_taus: int = 300, until: float = .1) -> np.ndarray:
    # Conservative estimate for meaningful τ values based on data.
    return np.geomspace(2/mmt.sample_rate, mmt.duration * until, num=n_taus)
