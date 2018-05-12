"""Parse output files from the Pendulum CNT-91 counter."""
import pandas as pd


def parse_txt(file_name: str) -> pd.Series:
    """Parse frequency measurement into a Pandas time series of freqs."""
    data = pd.read_table(file_name, squeeze=True, index_col=0, usecols=range(2))
    data.index = pd.to_datetime(data.index, unit='s',
                                origin=get_start_time(file_name))
    data.name = _get_info(file_name)
    return data


def get_start_time(file_name: str) -> pd.datetime:
    """Get the measurement starting time from a CNT-91 CSV file."""
    time_string = _get_info(file_name)[21:40]
    return pd.to_datetime(time_string)


def _get_info(file_name: str) -> str:
    with open(file_name) as file:
        info = file.readline().replace('\t', ' ').strip()
    return info
