""" Here we will build a dataLoader for the psd files."""
import cProfile
import pstats


from functools import cached_property

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path
from datetime import datetime
from typing import  Tuple, List, Optional, Union
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from config import settings, parse_datetime_strings
import numpy as np
import torch
import h5py
import json
import pyarrow.parquet as pq

class PSDDataset(Dataset):
    def __init__(self, filename: Union[Path, str], datetime_range: Tuple[datetime, datetime],
                 drop: List[str] = None, transform=None, label_transform=None) -> None:
        self.filename = filename
        self.transform = transform
        self.label_transform = label_transform
        self.drop = set(drop) if drop else set()

        # Open the parquet file once and reuse
        self.parquet_file = pq.ParquetFile(filename)
        
        # Efficiently select row groups within the datetime range
        start_timestamp = datetime_range[0].timestamp()
        end_timestamp = datetime_range[1].timestamp()
        self.keys = []
        for i in range(self.parquet_file.num_row_groups):
            row_group = self.parquet_file.metadata.row_group(i)
            min_time = row_group.column(0).statistics.min
            max_time = row_group.column(0).statistics.max
            if min_time <= end_timestamp and max_time >= start_timestamp:
                self.keys.append(i)

        self.initialize_dicts()
        
    def __len__(self) -> int:
        return len(self.keys)
    
    def initialize_dicts(self):
        # Read the required row group
        table = self.parquet_file.read_row_group(self.keys[0])
        sensor_names = table['sensor_name'].to_pylist()
        
        # Flatten the list if it is a list of lists
        if sensor_names and isinstance(sensor_names[0], list):
            sensor_names = [item for sublist in sensor_names for item in sublist]
        
        # Filter out the sensors not in drop
        sensor_names = {sensor for sensor in sensor_names if sensor not in self.drop}
        
        # Initialize sensor names
        self._sensor_names = list(sensor_names)

        # Initialize position names and axis names
        position_names = set()
        axis_names = set()
        for sensor in sensor_names:
            position_names.add(get_sensor_position(sensor))
            axis_names.add(get_sensor_axis(sensor))
        
        # Store them as attributes
        self._position_names = list(position_names)
        self._axis_names = list(axis_names)
        
        # Initialize mapping dictionaries
        self._mapping_dict = {sensor: i for i, sensor in enumerate(self._sensor_names)}
        self._mapping_axis_dict = {axis: i for i, axis in enumerate(self._axis_names)}
        self._mapping_position_dict = {position: i for i, position in enumerate(self._position_names)}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self.keys[index]
        
        # Read the required row group
        sample = self.parquet_file.read_row_group(key).to_pandas()
        
        psd = sample['ACC_psd'].values[0]
        sensor_name = sample['sensor_name'].values[0]
        
        # Flatten the list if it is a list of lists
        if len(sensor_name) > 0 and isinstance(sensor_name[0], list):
            sensor_name = [item for sublist in sensor_name for item in sublist]
        
        # Make sure that psd is 2D array
        psd = np.array(psd).reshape(len(sensor_name), -1)

        # Create boolean mask
        keep_rows = np.array([sensor not in self.drop for sensor in sensor_name])
        
        # Filter using boolean mask
        psd = psd[keep_rows]
        #sensor_name = [name for name, keep in zip(sensor_name, keep_rows) if keep]
        sensor_direnction = np.array([self._mapping_axis_dict[get_sensor_axis(sensor)] for sensor in sensor_name])
        sensor_position = np.array([self._mapping_position_dict[get_sensor_position(sensor)] for sensor in sensor_name])

        # One hot encode the sensor°s position and direction
        sensor_position = np.eye(len(self._position_names))[sensor_position]
        sensor_direnction = np.eye(len(self._axis_names))[sensor_direnction]

        # Convert to tensors
        psd = torch.from_numpy(psd)
        sensor_position = torch.from_numpy(sensor_position)
        sensor_direnction = torch.from_numpy(sensor_direnction)
        # Apply transformations if any
        if self.transform is not None:
            psd = self.transform(psd)
        if self.label_transform is not None:
            sensor_position = self.label_transform(sensor_position)
            sensor_direnction = self.label_transform(sensor_direnction)
        return psd, sensor_position, sensor_direnction

    
def get_sensor_axis(sensor_name: str):
    sensor_axis = sensor_name.split('_')[0][-1]
    return sensor_axis
def get_sensor_position(sensor_name: str):
    sensor_position = sensor_name.split('_')[1]
    return sensor_position

def load_parameter(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)  
         
def min_max_scaler(min, max):    
    def fn(x):
        return (x - min) / (max - min)
    return fn


def test():
    path_psd = Path(settings.default.path['processed_data']) / 'PSD_8192.parquet'
    path_metadata = Path(settings.default.path['processed_data']) / f'metadata_8192.json'
    #transformer min max scaler in torch
    parameter = load_parameter(path_metadata)
    min = parameter['min']
    max = parameter['max']


    training_range = parse_datetime_strings(settings.split.train)
    datetime_range = (training_range['start'], training_range['end'])
    # Create dataset
    dataset = PSDDataset(path_psd, datetime_range,transform=min_max_scaler(min,max), drop=['ACC1_X', 'ACC1_Y'])
    #compute min and max of the dataset for normalization using dat

    # Create the data loader


    dataloader = DataLoader(dataset, batch_size=128)
    
    return dataloader, parameter


if __name__ == '__main__':
    a,b=test()
    print(b)
