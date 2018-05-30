"""Parse output from various sources into `FreqSeries` objects."""
from typing import List
import pandas as pd

from .freq_series import FreqSeries


def fokus2_txt(file_name: str, session: str = None,
               drop_lines: List[int] = None) -> FreqSeries:
    """Parse frequency measurement done by the FOKUS2 Dual frequency comb.

    :param session: Measurement context. See `FreqSeries`'s `session` param.
    """
    data = pd.read_table(file_name, squeeze=True, index_col=0, usecols=[0, 1])
    if drop_lines is not None:
        data.drop(data.index[drop_lines], inplace=True)
    data.index = pd.to_datetime(data.index, unit='us')
    with open(file_name) as file:
        data.name = file.readline().strip()
    return FreqSeries(data, session=session)


def generic_freq_counter(
        file_name: str, session: str = None,
        time_unit: str = 's', original_freq: float = None) -> FreqSeries:
    """Parse a generic two-column counter file like (time, frequency).
    :param file_name: File to read from.
    :param time_unit: Which unit does the counter count time in? (s, ms, us, ns)
    """
    data = pd.read_table(file_name, squeeze=True, index_col=0, usecols=[0, 1])
    data.index = pd.to_datetime(data.index, unit=time_unit)
    return FreqSeries(data, session=session, original_freq=original_freq)


def pendulum_cnt91_txt(file_name: str, session: str = None) -> FreqSeries:
    """Parse frequency measurement done with a Pendulum CNT 91 counter.

    :param session: Measurement context. See `FreqSeries`'s `session` param.
    """
    def get_start_time(file_name: str) -> pd.datetime:
        """Get the measurement starting time from a CNT-91 CSV file."""
        time_string = _get_info(file_name)[21:40]
        return pd.to_datetime(time_string)

    def _get_info(file_name: str) -> str:
        with open(file_name) as file:
            info = file.readline().replace('\t', ' ').strip()
        return info

    data = pd.read_table(file_name, squeeze=True, index_col=0, usecols=[0, 1])
    data.index = pd.to_datetime(data.index, unit='s',
                                origin=get_start_time(file_name))
    data.name = _get_info(file_name)
    return FreqSeries(data, session=session)
