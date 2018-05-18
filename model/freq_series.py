"""Provides the FreqSeries class, an object wrapping counter measurements."""

import datetime
import pandas as pd

class FreqSeries:
    """A series of frequency measurements, as usually taken by a counter."""

    def __init__(self, data: pd.Series) -> None:
        """
        :raises ValueError: For non-equidistantly sampled data.
        """
        self._data = data
        self._sample_rate = self._calc_sample_rate()

    @property
    def data(self) -> pd.Series:
        return self._data

    @property
    def sample_rate(self) -> float:
        """The rate in Hz at which the freq. measurements have been taken."""
        return self._sample_rate

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
