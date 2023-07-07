""" Here we will build a dataLoader for the psd database"""
from typing import Tuple, List, Optional, Union
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import sqlite3
from torch.utils.data import Dataset, DataLoader
from config import settings, parse_datetime_strings
import pandas as pd 
import json
def min_max_scaler(min, max):    
    def fn(x):
        return (x - min) / (max - min)
    return fn

def get_sensor_axis(sensor_name: str):
    sensor_axis = sensor_name.split('_')[0][-1]
    return sensor_axis

def get_sensor_position(sensor_name: str):
    sensor_position = sensor_name.split('_')[1]
    return sensor_position

def custom_collate_fn(batch):
    # `batch` is a list of samples [(psd1, pos1, axis1), (psd2, pos2, axis2), ...]

    # Stack PSD tensors along a new leading dimension
    psds = torch.stack([item[0] for item in batch])

    # Stack positions and axes (or keep them as lists if that's what you want)
    positions = torch.tensor([item[1] for item in batch])
    axes = torch.tensor([item[2] for item in batch])

    # Return batched tensors
    return psds, positions, axes

# Use the custom collate function in the DataLoader


class PSDDataset(Dataset):
    def __init__(self, database_path: Union[Path, str], ts_start: datetime, ts_end: datetime,
                 drop: List[str] = None, transform=None, label_transform=None, preload: bool = False,
                 return_time: bool = False):
        """
        Dataset class for PSD data.

        :param database_path: Path to the database file
        :param ts_start: Timestamp for data start
        :param ts_end: Timestamp for data end
        :param drop: List of sensors to drop
        :param transform: Transformations to apply to PSD
        :param label_transform: Transformations to apply to labels
        :param preload: Preload data into memory
        :param return_time: Return timestamps with the data
        """
        self.database_path = database_path
        self.transform = transform
        self.label_transform = label_transform
        self.return_time = return_time
        self.drop = drop or []
        self.preload = preload
        self.conn = sqlite3.connect(str(self.database_path))
        self.keys = []

        ts_start, ts_end = ts_start.timestamp(), ts_end.timestamp()
        placeholders = ','.join('?' for _ in self.drop)
        query = f'''
            SELECT rowid FROM psd_data
            WHERE sensor_name NOT IN ({placeholders})
            AND time BETWEEN ? AND ?;
        '''
        parameters = tuple(self.drop) + (ts_start, ts_end)
        cursor = self.conn.execute(query, parameters)
        self.keys = [row[0] for row in cursor]

        self.initialize_dicts()

        if self.preload:
            self.data = self.preload_data()

    def __len__(self):
        return len(self.keys)

    def initialize_dicts(self):
        """Initialize sensor names, positions, and axes."""
        query = 'SELECT DISTINCT sensor_name FROM psd_data;'
        cursor = self.conn.execute(query)
        sensor_names = {row[0] for row in cursor if row[0] not in self.drop}

        axis_names = set(map(get_sensor_axis, sensor_names))
        position_names = set(map(get_sensor_position, sensor_names))

        self._sensor_names = list(sensor_names)
        self._axis_names = list(axis_names)
        self._position_names = list(position_names)

        self._mapping_axis_dict = {axis: i for i, axis in enumerate(self._axis_names)}
        self._mapping_position_dict = {position: i for i, position in enumerate(self._position_names)}

            

    
            
    
    def preload_data(self):
        """Preload all data into memory."""
        data = {}
        for index, key in enumerate(self.keys):
            query = '''
            SELECT * FROM psd_data
            WHERE rowid = ?;
            '''
            cursor = self.conn.execute(query, (key,))
            row = cursor.fetchone()
            
            if self.return_time:
                time = datetime.fromtimestamp(row[0])
            psd = torch.tensor(np.frombuffer(row[2], dtype=np.float32).copy())

            sensor_name = row[1]
            axis = get_sensor_axis(sensor_name)
            position = get_sensor_position(sensor_name)
            axis = self._mapping_axis_dict[axis]
            position = self._mapping_position_dict[position]
            
            if self.transform:
                psd = self.transform(psd)
            if self.label_transform:
                axis = self.label_transform(axis)
                position = self.label_transform(position)
            
            if self.return_time:
                data[key] = (psd, position, axis, time)
            else:
                data[key] = (psd, position, axis)
        return data


    def __getitem__(self, index: int):
        if self.preload:
            return self.data[self.keys[index]]
        else:
            key = self.keys[index]
            query = '''
            SELECT * FROM psd_data
            WHERE rowid = ?;
            '''
            cursor = self.conn.execute(query, (key,))
            row = cursor.fetchone()

            if self.return_time:
                time = datetime.fromtimestamp(row[0])
            # load psd as a torch with buffer data
            psd = np.frombuffer(row[2], dtype=torch.float32)

            sensor_name = row[1]
            axis = get_sensor_axis(sensor_name)
            position = get_sensor_position(sensor_name)
            axis = self._mapping_axis_dict[axis]
            position = self._mapping_position_dict[position]
            
            if self.transform:
                psd = self.transform(psd)
            if self.label_transform:
                axis = self.label_transform(axis)
                position = self.label_transform(position)
            
            if self.return_time:
                return psd, position, axis, time
            else:
                return psd, position, axis

        
from time import time
import matplotlib.pyplot as plt
def test():
    path_psd = Path(settings.default.path['processed_data']) / 'PSD8192.db'
    training_range = parse_datetime_strings(settings.split.train)
    init_time = time()
    dataset = PSDDataset(path_psd,
                         ts_start= training_range['start'],
                         ts_end = training_range['end'],
                         drop=['ACC1_X','ACC1_Y'],
                         preload=True)
    
    dataloader = DataLoader(dataset,batch_size=64,shuffle=2,
                            collate_fn=custom_collate_fn)
    set_time = time()
    for i,data in enumerate(dataloader):
        if i == 0 :
            plt.figure()
            plt.plot(data[0].T)
            plt.show()
            plt.close()
    final_time = time()
    print('init_time :', set_time - init_time)
    print('batch looping :', final_time - set_time)

def compute_min_max():
    path_psd = Path(settings.default.path['processed_data']) / 'PSD8192.db'
    training_range = parse_datetime_strings(settings.split.train)
    dataset_train = PSDDataset(path_psd,
                            ts_start= training_range['start'],
                            ts_end = training_range['end'],
                            drop=['ACC1_X','ACC1_Y'],
                            preload=True)
    min, max = dataset_train.compute_min_max()
    # save min max to json file transform min and max to float
    min = float(min)
    max = float(max)
    min_max_dict = {'min': min, 'max': max}
    min_max_path = Path(settings.default.path['processed_data']) / 'min_max_8192.json'

    with open(min_max_path, 'w') as f:
        json.dump(min_max_dict, f)




if __name__ =='__main__':
    compute_min_max()


    

    





# """ Here we will build a dataLoader for the psd files."""
# import cProfile
# import pstats


# from functools import cached_property

# import torch
# from torch.utils.data import Dataset, DataLoader
# import h5py
# from pathlib import Path
# from datetime import datetime
# from typing import  Tuple, List, Optional, Union
# from pathlib import Path
# from datetime import datetime
# import torch
# from torch.utils.data import Dataset, DataLoader
# import h5py
# from config import settings, parse_datetime_strings
# import numpy as np
# import torch
# import h5py
# import json
# import pyarrow.parquet as pq

# class PSDDataset(Dataset):
#     def __init__(self, filename: Union[Path, str], datetime_range: Tuple[datetime, datetime],
#                  drop: List[str] = None, transform=None, label_transform=None) -> None:
#         self.filename = filename
#         self.transform = transform
#         self.label_transform = label_transform
#         self.drop = set(drop) if drop else set()

#         # Open the parquet file once and reuse
#         self.parquet_file = pq.ParquetFile(filename)
        
#         # Efficiently select row groups within the datetime range
#         start_timestamp = datetime_range[0].timestamp()
#         end_timestamp = datetime_range[1].timestamp()
#         self.keys = []
#         for i in range(self.parquet_file.num_row_groups):
#             row_group = self.parquet_file.metadata.row_group(i)
#             min_time = row_group.column(0).statistics.min
#             max_time = row_group.column(0).statistics.max
#             if min_time <= end_timestamp and max_time >= start_timestamp:
#                 self.keys.append(i)

#         self.initialize_dicts()
#         #get shape of the data
        
        
#     def __len__(self) -> int:
#         return len(self.keys)
    
#     def initialize_dicts(self):
#         # Read the required row group
#         table = self.parquet_file.read_row_group(self.keys[0])
#         sensor_names = table['sensor_name'].to_pylist()
        
#         # Flatten the list if it is a list of lists
#         if sensor_names and isinstance(sensor_names[0], list):
#             sensor_names = [item for sublist in sensor_names for item in sublist]
        
#         # Filter out the sensors not in drop
#         sensor_names = {sensor for sensor in sensor_names if sensor not in self.drop}
        
#         # Initialize sensor names
#         self._sensor_names = list(sensor_names)

#         # Initialize position names and axis names
#         position_names = set()
#         axis_names = set()
#         for sensor in sensor_names:
#             position_names.add(get_sensor_position(sensor))
#             axis_names.add(get_sensor_axis(sensor))
        
#         # Store them as attributes
#         self._position_names = list(position_names)
#         self._axis_names = list(axis_names)
#         # sort them
#         self._position_names.sort()
#         self._axis_names.sort()
        
#         # Initialize mapping dictionaries
#         self._mapping_dict = {sensor: i for i, sensor in enumerate(self._sensor_names)}
#         self._mapping_axis_dict = {axis: i for i, axis in enumerate(self._axis_names)}
#         self._mapping_position_dict = {position: i for i, position in enumerate(self._position_names)}

#     def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         key = self.keys[index]
        
#         # Read the required row group
#         sample = self.parquet_file.read_row_group(key).to_pandas()
        
#         psd = sample['ACC_psd'].values[0]
#         sensor_name = sample['sensor_name'].values[0]
        
#         # Flatten the list if it is a list of lists
#         if len(sensor_name) > 0 and isinstance(sensor_name[0], list):
#             sensor_name = [item for sublist in sensor_name for item in sublist]
        
#         # Make sure that psd is 2D array
#         psd = np.array(psd).reshape(len(sensor_name), -1)

#         # Create boolean mask
#         keep_rows = np.array([sensor not in self.drop for sensor in sensor_name])
        
#         # Filter using boolean mask
#         psd = psd[keep_rows]
#         sensor_name = sensor_name[keep_rows]
#         #sensor_name = [name for name, keep in zip(sensor_name, keep_rows) if keep]
    
#         sensor_direction = np.array([self._mapping_axis_dict[get_sensor_axis(sensor)] \
#                                       for sensor in sensor_name])
#         sensor_position = np.array([self._mapping_position_dict[get_sensor_position(sensor)]\
#                                      for sensor in sensor_name])

#         # One hot encode the sensor°s position and direction
#         sensor_position = np.eye(len(self._position_names))[sensor_position]
#         sensor_direction = np.eye(len(self._axis_names))[sensor_direction]

#         # Convert to tensors
#         psd = torch.from_numpy(psd)
#         sensor_position = torch.from_numpy(sensor_position)
#         sensor_direction = torch.from_numpy(sensor_direction)
#         # Apply transformations if any
#         if self.transform is not None:
#             psd = self.transform(psd)
#         if self.label_transform is not None:
#             sensor_position = self.label_transform(sensor_position)
#             sensor_direction = self.label_transform(sensor_direction)
#         return psd, [sensor_position, sensor_direction]
    

    
# def get_sensor_axis(sensor_name: str):
#     sensor_axis = sensor_name.split('_')[0][-1]
#     return sensor_axis
# def get_sensor_position(sensor_name: str):
#     sensor_position = sensor_name.split('_')[1]
#     return sensor_position

# def load_parameter(path: Path) -> dict:
#     with open(path, 'r') as f:
#         return json.load(f)  
         
# def min_max_scaler(min, max):    
#     def fn(x):
#         return (x - min) / (max - min)
#     return fn


# def test():
#     path_psd = Path(settings.default.path['processed_data']) / 'PSD_8192.parquet'
#     path_metadata = Path(settings.default.path['processed_data']) / f'metadata_8192.json'
#     #transformer min max scaler in torch
#     parameter = load_parameter(path_metadata)
#     min = parameter['min']
#     max = parameter['max']


#     training_range = parse_datetime_strings(settings.split.train)
#     datetime_range = (training_range['start'], training_range['end'])
#     # Create dataset
#     dataset = PSDDataset(path_psd, datetime_range,transform=min_max_scaler(min,max), drop=['ACC1_X', 'ACC1_Y'])
#     #compute min and max of the dataset for normalization using dat

#     # Create the data loader


#     dataloader = DataLoader(dataset, batch_size=128)
    
#     return dataloader, parameter


# if __name__ == '__main__':
#     pass