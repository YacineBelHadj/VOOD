#%%
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from config import settings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import sys
EPS = sys.float_info.epsilon
from functools import wraps

def inplace_decorator(func):
    @wraps(func)
    def wrapper(*args, inplace=False, **kwargs):
        if inplace:
            func(*args, **kwargs)
            return args[0]
        else:
            return func(*args, **kwargs)
    return wrapper

def load_psd_from_hdf5(name_psd='PSD_8192.h5'):
    path_psd = Path(settings.default.path['project'])/Path(settings.default.path['processed_data']) / name_psd
    data = []
    with h5py.File(path_psd, 'r') as f:
        psd_group = f['PSDs']
        frequency = np.array(psd_group['frequency']).flatten()
        for sample in tqdm(psd_group.values()):
            try :
                sample_time = pd.Timestamp(sample.attrs['time'], unit='s')
            except:
                continue
            for sensor_group in sample.values():
                sensor_name = sensor_group.name.split('/')[-1]
                psd = np.array(sensor_group[()]).flatten()
                data.append([sample_time, sensor_name, psd])
    df = pd.DataFrame(data, columns=['time', 'sensor', 'psd'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    return df, frequency

@inplace_decorator
def remove_row(df, column, value):
    df = df.copy()
    if isinstance(value, list):
        for v in value:
            df = df[df[column] != v]
    else:
        print('value must be a list')
    return df

@inplace_decorator
def one_hot_encode(df:pd.Series):
    enc= OneHotEncoder()
    enc.fit(df.values.reshape(-1,1))
    one_hot=enc.transform(df.values.reshape(-1,1))
    return one_hot, enc

def normalize(df:pd.DataFrame,column='psd',inplace=False):
    arr=np.vstack(df[column].values)    
    func_normalize=lambda x: (np.array(x)-x.min())/(x.max()-x.min())
    return df[column].apply(func_normalize)

@inplace_decorator
def train_val_test_split(df, train_date= '2022-04-21',val_date='2022-04-25'):
        df = df.copy()
        df['train'] = df.index < train_date
        df['validation'] = (df.index>train_date) & (df.index<val_date) 
        return df

def get_array(df:pd.DataFrame, column='psd'):
    arr=np.vstack(df[column].values)
    return arr
#%%
if __name__ == '__main__':
    df,f= load_psd_from_hdf5()
    print(df.info())

    #%%
    df=remove_row(df,'sensor',['ACC1_X','ACC1_Y'])
    df['psd']=df['psd'].apply(lambda x:np.log(x+EPS))
    print(df.info())
    #%%
    df=train_val_test_split(df)
    df['psd']=normalize(df)
    encoded,encoder=one_hot_encode(df['sensor'])
    df['sensor']=encoded


    