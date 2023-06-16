import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import h5py
from scipy import signal
from VOOD.data import data_loader as dl
from config import settings
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager

EPS = sys.float_info.epsilon

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    filename='VOOD/logs/generate_data.log'
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
    
    # Apply the filter to the data
    data_filtered = signal.lfilter(b, a, data)
    

    # Step 3: Signal conditioning
    # Subtract the mean and divide by the standard deviation
    conditioned_data = (data_filtered - np.mean(data_filtered)) / np.std(data_filtered)
    rms= np.sqrt(np.mean(np.square(conditioned_data)))
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
    psds={}
    rms = {}
    f = None
    for sensor_name, sensor_data in data.items():
        processed_data, rms_sig = preprocess_vibration_data(sensor_data, fs, cut_off_freq)
        f, psd = apply_welch(processed_data, fs, psd_nperseg)
        psds[sensor_name] = psd
        rms[sensor_name] = rms_sig
    logging.info(f'Processed data at {dt}')
    return (i,dt.timestamp(),psds,f,rms)


def main():
    path_psd = Path(settings.default.path['processed_data']) / f'PSD_{psd_nperseg}.h5'
    with h5py.File(path_psd, 'w') as f:
        psd_group = f.create_group('PSDs')
        psd_group.attrs['fs'] = fs
        psd_group.attrs['frame(s)'] = frame_size.total_seconds()
        psd_group.attrs['step(s)'] = frame_step.total_seconds()
        psd_group.attrs['cut_off_freq'] = cut_off_freq
        psd_group.attrs['nperseg'] = psd_nperseg

        num_iter = int((end_time - start_time) / frame_step)
        args_list = [(i, start_time + i * frame_step) for i in range(num_iter)]
        
        with Pool() as pool, tqdm(total=num_iter) as pbar:
            results = pool.imap(process_iteration, args_list)

            for result in results:
                if result is None:
                    continue
                i, dt, psds, freq, rms = result
                sample_group = psd_group.create_group(str(i))
                psd_group[str(i)].attrs['time'] = dt
                for sensor_name, psd in psds.items():
                    sample_group.create_dataset(sensor_name, data=psd)
                    sample_group[sensor_name].attrs['rms'] = rms[sensor_name]
                pbar.update(1)

        psd_group.create_dataset('frequency', data=freq)
if __name__ == '__main__':
    main()