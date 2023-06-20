""" Here we will build a dataLoader for the psd files."""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path
from datetime import datetime
from typing import  Tuple
from typing import Tuple
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Dataset
import h5py
from config import settings, events
import numpy as np
from VOOD.c

class DatasetPSD(Dataset):
    """Dataset of PSDs, with the following parameters:
    filename: Path to the file containing the PSDs
    datetime_range: tuple of datetime objects, the first and last datetime of the dataset
    """
    def __init__(self, filename: Path or str, datetime_range: Tuple[datetime, datetime]) -> None:
        self.datetime_range = datetime_range
        self.filename = filename
        with h5py.File(filename, 'r') as f:
            psd_group = f['PSDs']
            self.keys = [key for key in psd_group.keys() if datetime_range[0] <= psd_group[key].attrs['time'] <= datetime_range[1]]
    
    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx) -> dict:
        key = self.keys[idx]
        with h5py.File(self.filename, 'r') as f:
            psd_group = f['PSDs']
            sample_group = psd_group[key]
            psds = {sensor_name: torch.tensor(sample_group[sensor_name][:]) for sensor_name in sample_group.keys() if sensor_name != 'frequency'}
            #freq = torch.tensor(sample_group['frequency'][:])
            #rms = {sensor_name: sample_group[sensor_name].attrs['rms'] for sensor_name in psds.keys()}
        return psds
    
def load_psd_parameter(filename:str|Path) -> dict:
    res =dict()
    with h5py.File(filename, 'r') as f:
        psd_group = f['PSDs']
        res ['fs'] = psd_group.attrs['fs']
        res ['frame_size'] = psd_group.attrs['frame(s)']
        res ['frame_step'] = psd_group.attrs['step(s)']
        res ['cut_off_freq'] = psd_group.attrs['cut_off_freq']
        res ['nperseg'] = psd_group.attrs['nperseg']
    return res
def load_frequnecy_array(filename:str|Path) -> np.ndarray:
    with h5py.File(filename, 'r') as f:
        psd_group = f['PSDs']
        return np.array(psd_group['frequency'])

def main():
    path_psd= Path(settings.default.path['processed_data'])/'PSD_8192.h5'
    print(load_psd_parameter(path_psd))
    print(load_frequnecy_array(path_psd))
    datetime_range = (datetime(2022, 1, 1), datetime(2019, 1, 2))
    print(events)

if __name__ == '__main__':
    main()

    #command line to check available packages
    #pip freeze > requirements.txt
    