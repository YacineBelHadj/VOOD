import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import logging
from scipy import signal
from VOOD.data import loader as dl
from config import settings
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from config import settings, parser
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import sys
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
    i, dt = args
    sensor = dl.Sensor(name='ACC', location='MO04', data_type='TDD', format='.tdms')
    loader = dl.DataLoader(sensor=sensor, time_step=state['frame_size'])
    data = loader.get_data(dt, dt + state['frame_size'])
    if data is None:
        logging.info(f'No data at {dt}')
        return None
    psds = []
    sensor_names = []
    rms = []
    f = None
    for sensor_name, sensor_data in data.items():
        processed_data, rms_sig = preprocess_vibration_data(sensor_data, state['fs'], state['cut_off_freq'])
        f, psd = apply_welch(processed_data, state['fs'], state['psd_nperseg'])
        # Remove the filter part of the signal from the PSD using a mask numpy
        filter_mask = np.logical_and(f < state['cut_off_freq'], f > 0.2)
        psd = psd[filter_mask]
        psd = np.log10(psd+EPS).astype(np.float32)
        f = f[filter_mask]
        psds.append(psd)
        sensor_names.append(sensor_name)
        rms.append(rms_sig)
    if not psds:
        logging.info(f'No data at {dt}')
        return None

    logging.info(f'Processed data at {dt}')
    return (i, dt.timestamp(), psds, sensor_names, f, rms)


def main():
    path_psd = Path(settings.default.path['processed_data'])
    database_path = path_psd / f'PSD{state["psd_nperseg"]}.db'  # Specify the path to your SQLite database file

    # Create an SQLite database connection
    conn = sqlite3.connect(database_path)

    # Define the SQL statement to create the table
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS psd_data (
        time REAL,
        sensor_name TEXT,
        psd BLOB,
        rms REAL
    );
    '''

    # Execute the SQL statement to create the table
    conn.execute(create_table_query)

    num_iter = int((time_range['end'] - time_range['start']) / state['frame_step'])
    args_list = [(i, time_range['start'] + i * state['frame_step']) for i in range(num_iter)]

    with ThreadPoolExecutor() as executor, tqdm(total=num_iter) as pbar:
        results = executor.map(process_iteration, args_list)

        # Create a list to hold the bulk insert data
        bulk_insert_data = []

        for result in results:
            if result is None:
                continue

            i, dt, psds, sensor_name, freq, rms = result

            for sensor_name, psd, rms_value in zip(sensor_name, psds, rms):
                # Serialize the PSD data before storing it in the SQLite table
                serialized_psd = psd.tobytes()

                # Append the data to the bulk insert list
                bulk_insert_data.append((dt, sensor_name, serialized_psd, rms_value))

            pbar.update(1)

        # Perform bulk insert using executemany()
        insert_query = '''
        INSERT INTO psd_data (time, sensor_name, psd, rms)
        VALUES (?, ?, ?, ?);
        '''
        conn.executemany(insert_query, bulk_insert_data)

        # Commit the changes
        conn.commit()

    # Close the SQLite database connection
    conn.close()

if __name__ == '__main__':
    main()
