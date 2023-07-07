import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import h5py
from scipy import signal
from VOOD.data import loader as dl
from config import settings
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
import pandas as pd
from tdms_utils import saveTDMS
from config import settings, parser
from pyarrow import parquet as pq
import pyarrow as pa
import json
import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
EPS = sys.float_info.epsilon

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    filename='logs/generate_data.log'
)

state = settings.psd.option1
time_range = parser['datetime'](settings.split.all)
state['frame_size'] = timedelta(seconds=state['frame_size'])
state['frame_step'] = timedelta(seconds=state['frame_step'])

def preprocess_vibration_data(data, sampling_frequency, lpf):
    # Step 1: Band-pass filtering
    # Design a Butterworth low-pass filter
    b, a = signal.butter(state['filter_order'], lpf, 'low', fs=sampling_frequency)
    rms= np.sqrt(np.mean(np.square(data)))

    # Apply the filter to the data
    data_filtered = signal.lfilter(b, a, data)
    

    # Step 3: Signal conditioning
    # Subtract the mean and divide by the standard deviation
    conditioned_data = (data_filtered - np.mean(data_filtered)) / np.std(data_filtered)
    return conditioned_data, rms

def apply_welch(sig, sr:int,nperseg:int,noverlap:int|None=None):
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx_den = signal.welch(sig, sr, nperseg=nperseg, noverlap=noverlap)
    return f, Pxx_den

def process_iteration(args):
    i,dt =args
    sensor = dl.Sensor(name='ACC', location='MO04', data_type='TDD', format='.tdms')
    loader = dl.DataLoader(sensor=sensor, time_step=state['frame_size'])
    data = loader.get_data(dt, dt + state['frame_size'])
    if data is None:
        logging.info(f'No data at {dt}')
        return None
    psds=[]
    sensor_names = []
    rms = []
    f = None
    for sensor_name, sensor_data in data.items():
        processed_data, rms_sig = preprocess_vibration_data(sensor_data, state['fs'], state['cut_off_freq'])
        f, psd = apply_welch(processed_data, state['fs'], state['psd_nperseg'])
        #remove the filter part of the signal from the PSD using mask numpy
        psd=psd[f<state['cut_off_freq']]
        f=f[f<state['cut_off_freq']]
        psds.append(psd)
        sensor_names.append(sensor_name)
        rms.append(rms_sig)
    if not psds:
        logging.info(f'No data at {dt}')
        return None
    


    logging.info(f'Processed data at {dt}')
    return (i,dt.timestamp(),psds,sensor_names,f,rms)


def main():
    path_psd = Path(settings.default.path['processed_data']) / f'PSD_{state["psd_nperseg"]}.avro'
    path_metadata = Path(settings.default.path['processed_data']) / f'metadata_{state["psd_nperseg"]}.json'
    
    # Define Avro schema
    schema = avro.schema.Parse(json.dumps({
        "type": "record",
        "name": "PSDData",
        "fields": [
            {"name": "time", "type": "double"},
            {"name": "ACC_psd", "type": {"type": "array", "items": "double"}},
            {"name": "ACC_rms", "type": "double"},
            {"name": "sensor_name", "type": "string"}
        ]
    }))
    
    num_iter = int((time_range['end'] - time_range['start']) / state['frame_step'])
    args_list = [(i, time_range['start'] + i * state['frame_step']) for i in range(num_iter)]
    
    min_values = {}
    max_values = {}
    freq = None
    
    with DataFileWriter(open(path_psd, "wb"), DatumWriter(), schema) as writer:
        with Pool() as pool, tqdm(total=num_iter) as pbar:
            results = pool.imap(process_iteration, args_list)
            
            for result in results:
                if result is None:
                    continue
                
                i, dt, psds, sensor_name, freq, rms = result
                
                data_to_add = [
                {
                    'time': float(dt),  # Ensuring it's a float (double in Avro terms)
                    'ACC_psd': psds[i].tolist() if isinstance(psds[i], np.ndarray) else psds[i],  # Convert to list if it's numpy array
                    'ACC_rms': float(rms[i]),  # Ensuring it's a float (double in Avro terms)
                    'sensor_name': str(sensor_name[i])  # Ensuring it's a string
                }for i in range(len(psds))]

                            
                # Writing records in Avro file
                for record in data_to_add:
                    writer.append(record)
                
                for sensor_name, psd in zip(sensor_name, psds):
                    min_values[sensor_name] = min(min_values.get(sensor_name, float('inf')), np.min(psd))
                    max_values[sensor_name] = max(max_values.get(sensor_name, float('-inf')), np.max(psd))
                
                pbar.update(1)
                
            if freq is not None:
                meta_data = {
                    'fs': state['fs'],
                    'frame(s)': state['frame_size'].total_seconds(),
                    'step(s)': state['frame_step'].total_seconds(),
                    'cut_off_freq': state['cut_off_freq'],
                    'nperseg': state['psd_nperseg'],
                    'freq': freq.tolist(),
                    'min': min,
                    'max': max
                }
                
                with open(path_metadata, 'w') as f:
                    json.dump(meta_data, f)

if __name__ == '__main__':
    main()

