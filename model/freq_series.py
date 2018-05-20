"""Provides the FreqSeries class, an object wrapping counter measurements."""

import datetime
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
        self._sample_rate = self._calc_sample_rate()

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
            local_sample_rate = times[sample + 1] - times[sample]
            if  abs(local_sample_rate - guess) > .01 * guess:
                raise ValueError("Non-uniform sample distance {}.".format(local_sample_rate))
        return 1/guess.total_seconds()
