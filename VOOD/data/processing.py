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


EPS = sys.float_info.epsilon

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    filename='logs/generate_data.log'
)

start_time = datetime(2022, 3, 30, 0, 0, 0)
end_time = datetime(2022, 6, 25, 0, 0, 0)
frame_size = timedelta(minutes=5)
frame_step = timedelta(minutes=4)
fs = 250
cut_off_freq = 50
psd_nperseg = 8192
filter_order = 4


def preprocess_vibration_data(data, sampling_frequency, lpf):
    # Step 1: Band-pass filtering
    # Design a Butterworth band-pass filter from 0.1 Hz to the Nyquist frequency
    b, a = signal.butter(filter_order, lpf / (sampling_frequency / 2))
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
    loader = dl.DataLoader(sensor=sensor, time_step=frame_size)
    data = loader.get_data(dt, dt + frame_size)
    if data is None:
        logging.info(f'No data at {dt}')
        return None
    psds=[]
    sensor_names = []
    rms = []
    f = None
    for sensor_name, sensor_data in data.items():
        processed_data, rms_sig = preprocess_vibration_data(sensor_data, fs, cut_off_freq)
        f, psd = apply_welch(processed_data, fs, psd_nperseg)
        #remove the filter part of the signal from the PSD using mask numpy
        psd=psd[f<cut_off_freq]
        f=f[f<cut_off_freq]
        psds.append(psd)
        sensor_names.append(sensor_name)
        rms.append(rms_sig)
    if not psds:
        logging.info(f'No data at {dt}')
        return None
    
    psds = np.stack(psds, axis=0).flatten()
    psds = np.log10(psds + EPS)
    sensor_names = np.array(sensor_names).flatten()
    rms = np.array(rms).flatten()
    logging.info(f'Processed data at {dt}')
    return (i,dt.timestamp(),psds,sensor_names,f,rms)

from pyarrow import parquet as pq
import pyarrow as pa
import json

def main():
    path_psd = Path(settings.default.path['processed_data']) / f'PSD_{psd_nperseg}.parquet'
    path_metadata = Path(settings.default.path['processed_data']) / f'metadata_{psd_nperseg}.json'
    num_iter = int((end_time - start_time) / frame_step)
    args_list = [(i, start_time + i * frame_step) for i in range(num_iter)]

    # Initializing the ParquetWriter
    # Please note: you might need to adjust schema based on your data
    schema = pa.schema([
        ('time', pa.float32()),
        ('ACC_psd', pa.list_(pa.float32())),
        ('ACC_rms', pa.list_(pa.float32())),
        ("sensor_name", pa.list_(pa.string())),
    ])
    min = float('-inf')
    max = float('inf')

    with pq.ParquetWriter(path_psd, schema) as writer:
        with Pool() as pool, tqdm(total=num_iter) as pbar:
            results = pool.imap(process_iteration, args_list)

            for result in results:
                if result is None:
                    continue
                i, dt, psds, sensor_name, freq, rms = result


                # Here we create a pandas DataFrame and then convert it to a PyArrow Table
                data = {'time': dt}
                data.update({'sensor_name': sensor_name})
                data.update({'ACC_psd': psds})
                data.update({'ACC_rms': rms})
                df = pd.DataFrame()
                df['sensor_name'] = [sensor_name]
                df['ACC_psd'] = [psds]
                df['ACC_rms'] = [rms]
                df['time'] = dt
                table = pa.Table.from_pandas(df, schema=schema)

                min = np.max([min, np.min(psds)])
                max = np.min([max, np.max(psds)])

                # Now we write the table into a Parquet file
                writer.write_table(table)
                pbar.update(1)
        # add metadata
        meta_data = {'fs': fs,
                        'frame(s)': frame_size.total_seconds(),
                        'step(s)': frame_step.total_seconds(),
                        'cut_off_freq': cut_off_freq,
                        'nperseg': psd_nperseg,
                        'freq': freq.tolist(),
                        'min': min,
                        'max': max}


    with open(path_metadata, 'w') as f:
        json.dump(meta_data, f)



if __name__ == '__main__':
    main()


# def main():
#     path_psd = Path(settings.default.path['processed_data']) / f'PSD_{psd_nperseg}.h5'
#     with h5py.File(path_psd, 'w') as f:
#         psd_group = f.create_group('PSDs')
#         psd_group.attrs['fs'] = fs
#         psd_group.attrs['frame(s)'] = frame_size.total_seconds()
#         psd_group.attrs['step(s)'] = frame_step.total_seconds()
#         psd_group.attrs['cut_off_freq'] = cut_off_freq
#         psd_group.attrs['nperseg'] = psd_nperseg

#         num_iter = int((end_time - start_time) / frame_step)
#         args_list = [(i, start_time + i * frame_step) for i in range(num_iter)]
        
#         with Pool() as pool, tqdm(total=num_iter) as pbar:
#             results = pool.imap(process_iteration, args_list)

#             for result in results:
#                 if result is None:
#                     continue
#                 i, dt, psds, freq, rms = result
#                 sample_group = psd_group.create_group(str(i))
#                 psd_group[str(i)].attrs['time'] = dt
#                 for sensor_name, psd in psds.items():
#                     sample_group.create_dataset(sensor_name, data=psd)
#                     sample_group[sensor_name].attrs['rms'] = rms[sensor_name]
#                 pbar.update(1)

#         f.create_dataset('frequency', data=freq)
# if False:
