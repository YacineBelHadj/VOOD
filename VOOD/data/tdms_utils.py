from typing import Union
from pathlib import Path
from nptdms import TdmsWriter, TdmsFile
from datetime import datetime
import warnings
import numpy as np

def readTDMS(path: Union[Path, str]):
    """ Read TDMS file
    Parameters
    ----------
    path : Union[Path, str]
        Path to the TDMS file
    Returns
    -------
    dict_signal : dict
        Dictionary containing signals from the file
    """
    if isinstance(path, Path):
        path = str(path)

    try:
        data = TdmsFile(path)
    except (FileNotFoundError, ValueError) as error:
        warnings.warn(f'FAILED IMPORT: {str(error)}', UserWarning)
        return {}

    if not data.groups():
        warnings.warn(f'FAILED IMPORT: No TDMS group found in file: {path}', UserWarning)
        return {}

    signals = [channel.data for group in data.groups() for channel in group.channels()]
    channels = [channel.name for group in data.groups() for channel in group.channels()]

    dict_signal = dict(zip(channels, signals))
    return dict_signal

def saveTDMS(path: Union[Path, str], dict_signal: dict, group_name: str = 'group'):
    """ Save TDMS file
    Parameters
    ----------
    path : Union[Path, str]
        Path to the TDMS file
    dict_signal : dict
        Dictionary containing signals to be saved
    group_name : str, optional
        Name of the group, by default 'group'
    """
    if isinstance(path, Path):
        path = str(path)

    try:
        data = TdmsFile(path)
    except (FileNotFoundError, ValueError) as error:
        warnings.warn(f'FAILED IMPORT: {str(error)}', UserWarning)
        return {}

    if not data.groups():
        warnings.warn(f'FAILED IMPORT: No TDMS group found in file: {path}', UserWarning)
        return {}

    signals = [channel.data for group in data.groups() for channel in group.channels()]
    channels = [channel.name for group in data.groups() for channel in group.channels()]

    dict_signal = dict(zip(channels, signals))
    with TdmsWriter(path) as tdms_writer:
        tdms_writer.write_segment([dict_signal], group_name=group_name)
    return dict_signal

def append_dict(dict1: dict, dict2: dict):
    """Append two dictionaries
    Parameters
    ----------
    dict1 : dict
        First dictionary
    dict2 : dict
        Second dictionary
    Returns
    -------
    dict1 : dict
        Appended dictionary
    """
    keys_dict1 = set(dict1.keys())
    keys_dict2 = set(dict2.keys())

    if keys_dict1 != keys_dict2:
        print("The keys of the dictionaries are not similar.")
        print("Keys in dict1:", keys_dict1)
        print("Keys in dict2:", keys_dict2)
        return dict1

    for key, value in dict2.items():
        dict1[key] = np.append(dict1[key], value)

    return dict1


def datetime_to_path(path_root: Union[Path, str], dt: datetime, file_type: str = 'tdms'):
    """Convert datetime to path
    Parameters
    ----------
    path_root : Union[Path, str]
        Root path for the file
    dt : datetime
        Datetime to be converted to path
    file_type : str, optional
        File type extension, by default 'tdms'
    Returns
    -------
    path : Path
        Path of the file
    """
    path_root = Path(path_root) if isinstance(path_root, str) else path_root
    path = path_root.joinpath(dt.strftime('%Y/%m/%d/%Y%m%d_%H%M%S')).with_suffix(file_type)
    return path
