from VOOD.data.tdms_utils import readTDMS, append_dict, datetime_to_path
from dataclasses import dataclass
from typing import Union
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config import settings

@dataclass(frozen=True)
class Sensor:
    """Sensor class to store sensor information"""
    name: str = 'ACC'
    location: str = 'MO04'
    data_type: str = 'TDD'
    format: str = '.tdms'


@dataclass
class DataLoader:
    """Data loader class to load data from a file"""
    sensor: Sensor = None
    time_step: timedelta = timedelta(minutes=1)

    def __post_init__(self):
        """Load config file and set data root and path"""
        data_root = Path(settings.default.path['raw_data'])
        if not data_root.exists():
            raise ValueError(f'Invalid data root: {data_root}')
        self.path = data_root / self.sensor.location / self.sensor.data_type / f'{self.sensor.data_type}_{self.sensor.name}'

    def _load_single(self, dt: datetime) -> dict:
        """Load data from a single file"""
        path = datetime_to_path(self.path, dt, self.sensor.format)
        if not path.exists():
            print(f'No data at {path}')
            return None
        return readTDMS(path)

    def _load(self, start, end):
        """Load data from start to end"""
        data = {}
        dt = start
        while dt < end:
            data_temp = self._load_single(dt)
            if data_temp is None:
                return None 
            if len(data) == 0:
                data = data_temp
            else:
                data= append_dict(data,data_temp)
            dt += self.time_step
        return data

    def get_data(self, start: Union[datetime, str], end: Union[datetime, str, None] = None) -> Union[dict, None]:
        """Get data from start to end"""
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if end is None:
            return self._load_single(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        if end < start:
            raise ValueError(f'End date {end} is before start date {start}')
        delta = end - start

        if delta > timedelta(days=1):
            raise ValueError(f'Cannot load more than one day at a time. Got {delta.days} days')

        return self._load(start, end)
    


if __name__ == '__main__':
    sensor = Sensor(name='ACC', location='MO04', data_type='TDD', format='.tdms')
    loader = DataLoader(sensor=sensor)
    data = loader.get_data('2022-04-20 00:00:00', '2022-04-20 00:10:00')
    print(data)
