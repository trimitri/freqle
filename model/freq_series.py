"""Provides the FreqSeries class, an object wrapping counter measurements."""

import datetime
import pandas as pd

class FreqSeries:
    """A series of frequency measurements, as usually taken by a counter.

    Indices (times) must be strictly monotonic.
    """

    def __init__(self, data: pd.Series) -> None:
        """
        :raises ValueError: For non-equidistantly sampled data.
        """
        if not data.index.is_monotonic_increasing:
            raise ValueError("Times are not monotonically increasing.")

        self._data: pd.Series = data
        self._sample_rate = self._calc_sample_rate()

    @property
    def data(self) -> pd.Series:
        return self._data

    @property
    def duration(self) -> float:
        """Duration of the measurement in seconds."""
        idx = self._data.index
        return (idx[-1] - idx[0]).total_seconds()

    @property
    def sample_rate(self) -> float:
        """The rate in Hz at which the freq. measurements have been taken."""
        return self._sample_rate

    def trim(self, start: int = None, end: int = None) -> None:
        """Trim some values from start and/or end (in-place operation).

        Does nothing, if neither `start` nor `end` are specified.
        """
        if start:
            self._data = self._data.iloc[start:]
        if end:
            self._data = self._data.iloc[:-end]

    def _calc_sample_rate(self) -> float:
        """
        :returns: Sample rate in Hz.
        :raises ValueError: The sample rate is not uniform.
        """
        times = self._data.index
        guess: datetime.timedelta = times[1] - times[0]
        for sample in range(len(times) - 1):
            if times[sample + 1] - times[sample] != guess:
                raise ValueError("Non-uniform sample rate.")
        return 1/guess.total_seconds()
