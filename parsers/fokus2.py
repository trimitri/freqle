"""Parse output files from the FOKUS-II Payload."""
from typing import List
import pandas as pd

from ..model.freq_series import FreqSeries


def parse_txt(file_name: str, session: str = None,
              drop_lines: List[int] = None) -> FreqSeries:
    """Parse frequency measurement into a Pandas time series of freqs.

    :param session: Measurement context. See `FreqSeries`'s `session` param.
    """
    data = pd.read_table(file_name, squeeze=True, index_col=0, usecols=[0, 1])
    if drop_lines is not None:
        data.drop(data.index[drop_lines], inplace=True)
    data.index = pd.to_datetime(data.index, unit='us')
    with open(file_name) as file:
        data.name = file.readline().strip()
    return FreqSeries(data, session=session)
