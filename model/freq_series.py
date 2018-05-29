"""Provides the FreqSeries class, an object wrapping counter measurements."""

import datetime
from typing import Tuple
import numpy as np
import pandas as pd

class FreqSeries:
    """A series of frequency measurements, as usually taken by a counter.

    Indices (times) must be strictly monotonic.
    """

    def __init__(self, data: pd.Series, original_freq: float = None,
                 session: str = None) -> None:
        """
        :param data: The measured frequencies.
        :param original_freq: The frequency that the DUT was actually operated
                        at. For physical (and numerical) reasons, the actual
                        frequency fluctuation of the oscillator is often scaled
                        down to the <= GHz range when measuring. This parameter
                        specifies the original frequency, that the fluctuations
                        have to be related to.
                        If `None` is given, absence of frequency scaling
                        (direct measurement!) is implied.
        :param session: Label for indicating the context of the measurement.
                        Especially useful when different setups are shown in
                        one plot. A separate session shoud be assigned for
                        every change in the experimental setup.
        :raises ValueError: For non-equidistantly sampled data.
        """
        if not data.index.is_monotonic_increasing:
            raise ValueError("Times are not monotonically increasing.")

        self.org_freq = original_freq
        self.session = session
        self._data: pd.Series = data

        rate, regularity = self._analyze_sample_rate()
        self._rate: float = rate
        """Median sample rate of the data in Hz.
        """
        self._regularity: float = regularity
        """How much does the sample rate deviate?

        max((max_rate / median_rate), (median_rate / min_rate))
        """

    @property
    def data(self) -> pd.Series:
        return self._data

    @property
    def duration(self) -> float:
        """Duration of the measurement in seconds."""
        idx = self._data.index
        delta: datetime.timedelta = idx[-1] - idx[0]
        return delta.total_seconds()

    @property
    def sample_rate(self) -> float:
        """The rate in Hz at which the freq. measurements have been taken."""
        return self._rate

    @property
    def sampling_regularity(self) -> float:
        return self._regularity

    def trim(self, start: int = None, end: int = None) -> None:
        """Trim some values from start and/or end (in-place operation).

        Does nothing, if neither `start` nor `end` are specified.
        """
        if start is not None:
            self._data = self._data.iloc[start:]
        if end is not None:
            self._data = self._data.iloc[:-end]

    def _analyze_sample_rate(self) -> Tuple[float, float]:
        """
        :returns: (median rate in Hz, rate regularity)
        :raises ValueError: The sample rate is not uniform.
        """
        times = np.diff([date.timestamp() for date in self._data.index])
        median = np.median(times)
        uniformity = max(np.max(times) / median, median / np.min(times))
        return (1/float(median), float(uniformity))
