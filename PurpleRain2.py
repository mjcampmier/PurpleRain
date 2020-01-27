'''
Author: Mark Campmier
Github/Twitter: @mjcampmier
Last Edit: 27 Jan 2020
'''
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.options import Options as FOptions
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from time import sleep
from scipy.io import savemat
from scipy import stats
from datetime import datetime, timedelta
from dateutil import tz
from matplotlib import cm
from sklearn import metrics
from folium.plugins import FloatImage
from folium.plugins import TimestampedGeoJson
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.colors as col
import calendar as cal
import h5py as h5
import math
import glob
import warnings
import itertools
import jdutil as jd
import folium
import imageio
import numpy as np
import pandas as pd
import requests
import os
import csv

warnings.filterwarnings("ignore")

def wrapto360(angles):
    '''
    Wraps negative angles [-180, 180] to
    [0, 360].
    '''
    for i in range(0,len(angles)):
        angle = angles[i]
        if angle < 0:
            angle = 360 + angle
            angle = np.absolute(angle)
        else:
            angle = angle
        angles[i] = np.absolute(angle)
    return angles

def isFloat(string):
    try:
        float(string)
        return True
    except:
        return False

def is_numf(x):
    '''
    Function to determine if input can be converted to float, 
    if not, returned as Numpy nan.
    '''
    try:
        float(x)
    except ValueError:
        y = np.nan
    except IndexError:
        y = np.nan
    else:
        y = float(x)
    return y

def is_numi(x):
    '''
    Function to determine if input can be converted to integer, 
    if not, returned as Numpy nan.
    '''
    try:
        int(x)
    except ValueError:
        y = np.nan
    except IndexError:
        y = np.nan
    else:
        y = int(x)
    return y

def build_dir(dir_path):
    '''
    Function to build folder in specified directory path.
    '''
    try:
        os.makedirs(dir_path)
        #print('New folder created!')
    except:
        stg = 'Folder already exists'
        #print('Folder already exists.')
    return dir_path

def download_database():
    try:
        r = requests.get('https://www.purpleair.com/json')
    except:
        print('Fatal Error: Make sure you have an internet connection, wait and try again.')
        return np.nan
    else:
        all_pa = dict(r.json())
        df_pa = pd.DataFrame(all_pa['results'])
        df_pa.drop(['AGE','A_H','DEVICE_LOCATIONTYPE',
                    'Flag','Hidden',
                    'ID','LastSeen','Ozone1',
                    'PM2_5Value','ParentID',
                    'Stats','Type','Voc',
                    'humidity','isOwner',
                    'pressure', 'temp_f',
                   ], axis=1, inplace=True)
        print('Database succesfully scraped.')
        return df_pa

def sensor_metadata(df_pa, sensor):
    sensor_A = sensor
    sensor_B = sensor+' B'
    try:
        primary_id_A = str(df_pa.THINGSPEAK_PRIMARY_ID[df_pa.Label == sensor_A].item())
        primary_key_A = df_pa.THINGSPEAK_PRIMARY_ID_READ_KEY[df_pa.Label == sensor_A].item()
        secondary_id_A = str(df_pa.THINGSPEAK_SECONDARY_ID[df_pa.Label == sensor_A].item())
        secondary_key_A = df_pa.THINGSPEAK_SECONDARY_ID_READ_KEY[df_pa.Label == sensor_A].item()

        primary_id_B = str(df_pa.THINGSPEAK_PRIMARY_ID[df_pa.Label == sensor_B].item())
        primary_key_B = df_pa.THINGSPEAK_PRIMARY_ID_READ_KEY[df_pa.Label == sensor_B].item()
        secondary_id_B = str(df_pa.THINGSPEAK_SECONDARY_ID[df_pa.Label == sensor_B].item())
        secondary_key_B = df_pa.THINGSPEAK_SECONDARY_ID_READ_KEY[df_pa.Label == sensor_B].item()

        ID = [primary_id_A, secondary_id_A, primary_id_B, secondary_id_B]
        KEYS = [primary_key_A, secondary_key_A, primary_key_B, secondary_key_B]
    except ValueError:
        print('Name not found. Please check your PurpleAir registration for: ', sensor)
        ID = [np.nan, np.nan, np.nan, np.nan]
        KEYS = [np.nan, np.nan, np.nan, np.nan]
    return ID, KEYS

def download_request(ID, KEY, SD, ED, fname, down_dir):
    SD_list = SD.split('-')
    ED_list = ED.split('-')
    sd = pd.Timestamp(int(SD_list[0]), 
                      int(SD_list[1]), 
                      int(SD_list[2]))
    ed = pd.Timestamp(int(ED_list[0]), 
                      int(ED_list[1]), 
                      int(ED_list[2]))
    date_ind = pd.date_range(sd, ed, freq='5D')
    dir_path = build_dir(down_dir+'/'+fname)
    for i in range(1, len(date_ind)):
        if i != len(date_ind)-1:
            SDi = str(date_ind[i]).split(' ')[0]
            EDi = str(date_ind[i+1]).split(' ')[0]
            url = 'https://thingspeak.com/channels/'+            ID+'/feed.csv?api_key='+            KEY+'&offset=0&average=0&            round=2&start='+SDi+'%2000:00            :SS&end='+EDi+'%2000:00:00'
            r = requests.get(url, allow_redirects=True)
            open(SDi+'.csv', 'wb').write(r.content)
            os.rename(SDi+'.csv', dir_path+'/'+SDi+'.csv')
    flist = glob.glob(dir_path+'/*.csv')
    full_df = pd.read_csv(flist[0])
    for i in range(1, len(flist)):
        temp_df = pd.read_csv(flist[i])
        full_df = pd.concat([full_df, temp_df])
    full_df = full_df.sort_values(by=['created_at'])
    full_name = down_dir+'/'+fname+'.csv'
    full_df.to_csv(full_name, index=False)
    return full_name
    
def assign_field_names(fname, fields):
    df_csv = pd.read_csv(fname)
    if len(df_csv.columns) != len(fields):
        del fields[1]
        df_csv.columns = fields
    else:
        df_csv.columns = fields
    df_csv.to_csv(fname,index=False)

def download_sensor(sensor, SD, ED, down_dir, db = []):
    if len(db) == 0:
        df_pa = download_database()
    else:
        df_pa = db
    ID, KEYS = sensor_metadata(df_pa, sensor)
    sensor_A = sensor.replace(' ','_')
    sensor_B = sensor_A+'_B'
    sd = SD.replace('-','_')
    ed = ED.replace('-','_')
    fname = ['Primary_'+sensor_A+'_'+sd+'_'+ed,
            'Secondary_'+sensor_A+'_'+sd+'_'+ed,
            'Primary_'+sensor_B+'_'+sd+'_'+ed,
            'Secondary_'+sensor_B+'_'+sd+'_'+ed]
    fields_primary_A = ["created_at","entry_id","PM1.0_CF1_ug/m3",
                        "PM2.5_CF1_ug/m3","PM10.0_CF1_ug/m3","UptimeMinutes",
                        "ADC","Temperature_F","Humidity_%","PM2.5_ATM_ug/m3"]
    fields_secondary_A = ["created_at","entry_id",
                          ">=0.3um/dl",">=0.5um/dl",">1.0um/dl",">=2.5um/dl",
                          ">=5.0um/dl",">=10.0um/dl","PM1.0_ATM_ug/m3","PM10_ATM_ug/m3"]
    fields_primary_B = ["created_at","entry_id","PM1.0_CF1_ug/m3",
                        "PM2.5_CF1_ug/m3","PM10.0_CF1_ug/m3","UptimeMinutes",
                        "RSSI_dbm","Pressure_hpa","IAQ","PM2.5_ATM_ug/m3"]
    fields_secondary_B = ["created_at","entry_id",
                          ">=0.3um/dl",">=0.5um/dl",">1.0um/dl",">=2.5um/dl",
                          ">=5.0um/dl",">=10.0um/dl","PM1.0_ATM_ug/m3","PM10_ATM_ug/m3"]
    fields = [fields_primary_A, fields_secondary_A, 
              fields_primary_B, fields_secondary_B]
    for i in range(0, len(ID)):
        full_name  = download_request(ID[i], KEYS[i], SD, ED, fname[i], down_dir)
        assign_field_names(full_name, fields[i])
    print('Succesfully downloaded '+str(sensor))

def download_list(sensor_list_file, SD, ED, dir_name, hdfname, tz):
    build_dir(dir_name)
    sensor_list = pd.read_csv(sensor_list_file, header=None)
    sensor_list = sensor_list.iloc[:,0]
    df_db = download_database()
    for i in range(0, len(sensor_list)):
        download_sensor(sensor_list[i], SD, ED, dir_name, db = df_db)
    names = downloaded_file_list(dir_name+'/', sensor_list.tolist())
    build_hdf(names,hdfname, tz, dir_name)
    hdf5_to_mat(hdfname+'.h5')
    print('Succesfully downloaded all sensors.')

def time_master(df, col_names, tzstr):
    '''
    Aligns time-series data from input dataframe, 
    assigns datetime as index, converts UTC to local time
    based on these timezones (http://pytz.sourceforge.net/#country-information).
    Returns 1-hr Mean, Median, Standard Deviation, 
    and Max for all Mass, Count, and Meteorological Values
    '''
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz(tzstr)
    t = df.Time
    time_ind = []
    for i in range(0,len(t)):
        tstr = t[i].replace(' UTC','')
        time_ind.append(pd.to_datetime(tstr))
        time_ind[i] = time_ind[i].replace(tzinfo = from_zone)
        time_ind[i] = time_ind[i].astimezone(to_zone)
    df2 = pd.DataFrame(df.iloc[:,1:].values, index=time_ind)
    df_mean = df2.resample('60T').apply(np.nanmean)
    df_med  = df2.resample('60T').apply(np.nanmedian)
    df_std  = df2.resample('60T').apply(np.nanstd)
    df_max  = df2.resample('60T').apply(np.max)
    col_names = col_names.values.tolist()[1:]
    df_mean.columns = [s + '_mean' for s in col_names]
    df_med.columns  = [s + '_med' for s in col_names]
    df_std.columns  = [s + '_std' for s in col_names]
    df_max.columns  = [s + '_max' for s in col_names]
    df_summary = pd.concat([df_mean, df_med, df_std, df_max], axis=1)
    return df_summary

def file_hdf(sname):
    '''
    Intializes hdf file, and builds unique sensor list. 
    Returns writtable hdf file and unique sensor list.
    '''
    h5file = h5.File(sname+'.h5','w')
    return h5file

def fill_hdf(h5file, sensor, dfsum, dfpa, dfsa, dfpb, dfsb):
    '''
    First builds hdf file directory, then fills in the 
    values from [Primary A], [Secondary A], [Primary B],
    [Secondary B] dataframes. Flushes file to close,
    and returns h5file object.
    '''
    h = 'PM25'
    t_pa = h5file.create_group(sensor+'/A/P_Time')
    t_pb = h5file.create_group(sensor+'/B/P_Time') 
    t_sa = h5file.create_group(sensor+'/A/S_Time') 
    t_sb = h5file.create_group(sensor+'/B/S_Time')
    time = h5file.create_group(sensor+'/Time')
    a_raw = h5file.create_group(sensor+'/A/PM_Raw')
    a_cf = h5file.create_group(sensor+'/A/PM_CF')
    a_r_p1 = h5file.create_group(sensor+'/A/PM_Raw/PM1')
    a_r_p1_data = h5file.create_group(sensor+'/A/PM_Raw/PM1/Data')
    a_r_p1_sub = h5file.create_group(sensor+'/A/PM_Raw/PM1/Subsampled')
    a_r_p25 = h5file.create_group(sensor+'/A/PM_Raw/PM25')
    a_r_p25_data = h5file.create_group(sensor+'/A/PM_Raw/PM25/Data')
    a_r_p25_sub = h5file.create_group(sensor+'/A/PM_Raw/PM25/Subsampled')
    a_r_p10 = h5file.create_group(sensor+'/A/PM_Raw/PM10')
    a_r_p10_data = h5file.create_group(sensor+'/A/PM_Raw/PM10/Data')
    a_r_p10_sub = h5file.create_group(sensor+'/A/PM_Raw/PM10/Subsampled')
    a_cf_p1 = h5file.create_group(sensor+'/A/PM_CF/PM1')
    a_cf_p1_data = h5file.create_group(sensor+'/A/PM_CF/PM1/Data')
    a_cf_p1_sub = h5file.create_group(sensor+'/A/PM_CF/PM1/Subsampled')
    a_cf_p25 = h5file.create_group(sensor+'/A/PM_CF/PM25')
    a_cf_p25_data = h5file.create_group(sensor+'/A/PM_CF/PM25/Data')
    a_cf_p25_sub = h5file.create_group(sensor+'/A/PM_CF/PM25/Subsampled')
    a_cf_p10 = h5file.create_group(sensor+'/A/PM_CF/PM10')
    a_cf_p10_data = h5file.create_group(sensor+'/A/PM_CF/PM10/Data')
    a_cf_p10_sub = h5file.create_group(sensor+'/A/PM_CF/PM10/Subsampled')
    a_temp = h5file.create_group(sensor+'/A/Temperature')
    a_temp_data = h5file.create_group(sensor+'/A/Temperature/Data')
    a_temp_sub = h5file.create_group(sensor+'/A/Temperature/Subsampled')
    a_rh = h5file.create_group(sensor+'/A/Relative_Humidity')
    a_rh_data = h5file.create_group(sensor+'/A/Relative_Humidity/Data')
    a_rh_sub = h5file.create_group(sensor+'/A/Relative_Humidity/Subsampled')
    a_counts = h5file.create_group(sensor+'/A/Counts')
    a_c_03 = h5file.create_group(sensor+'/A/Counts/PM03')
    a_c_03_data = h5file.create_group(sensor+'/A/Counts/PM03/Data')
    a_c_03_sub = h5file.create_group(sensor+'/A/Counts/PM03/Subsampled')
    a_c_05 = h5file.create_group(sensor+'/A/Counts/PM05')
    a_c_05_data = h5file.create_group(sensor+'/A/Counts/PM05/Data')
    a_c_05_sub = h5file.create_group(sensor+'/A/Counts/PM05/Subsampled')
    a_c_1 = h5file.create_group(sensor+'/A/Counts/PM1')
    a_c_1_data = h5file.create_group(sensor+'/A/Counts/PM1/Data')
    a_c_1_sub = h5file.create_group(sensor+'/A/Counts/PM1/Subsampled')
    a_c_25 = h5file.create_group(sensor+'/A/Counts/PM25')
    a_c_25_data = h5file.create_group(sensor+'/A/Counts/PM25/Data')
    a_c_25_sub = h5file.create_group(sensor+'/A/Counts/PM25/Subsampled')
    a_c_10 = h5file.create_group(sensor+'/A/Counts/PM10')
    a_c_10_data = h5file.create_group(sensor+'/A/Counts/PM10/Data')
    a_c_10_sub = h5file.create_group(sensor+'/A/Counts/PM10/Subsampled')
    b_raw = h5file.create_group(sensor+'/B/PM_Raw')
    b_cf = h5file.create_group(sensor+'/B/PM_CF')
    b_r_p1 = h5file.create_group(sensor+'/B/PM_Raw/PM1')
    b_r_p1_data = h5file.create_group(sensor+'/B/PM_Raw/PM1/Data')
    b_r_p1_sub = h5file.create_group(sensor+'/B/PM_Raw/PM1/Subsampled')
    b_r_p25 = h5file.create_group(sensor+'/B/PM_Raw/PM25')
    b_r_p25_data = h5file.create_group(sensor+'/B/PM_Raw/PM25/Data')
    b_r_p25_sub = h5file.create_group(sensor+'/B/PM_Raw/PM25/Subsampled')
    b_r_p10 = h5file.create_group(sensor+'/B/PM_Raw/PM10')
    b_r_p10_data = h5file.create_group(sensor+'/B/PM_Raw/PM10/Data')
    b_r_p10_sub = h5file.create_group(sensor+'/B/PM_Raw/PM10/Subsampled')
    b_cf_p1 = h5file.create_group(sensor+'/B/PM_CF/PM1')
    b_cf_p1_data = h5file.create_group(sensor+'/B/PM_CF/PM1/Data')
    b_cf_p1_sub = h5file.create_group(sensor+'/B/PM_CF/PM1/Subsampled')
    b_cf_p25 = h5file.create_group(sensor+'/B/PM_CF/PM25')
    b_cf_p25_data = h5file.create_group(sensor+'/B/PM_CF/PM25/Data')
    b_cf_p25_sub = h5file.create_group(sensor+'/B/PM_CF/PM25/Subsampled')
    b_cf_p10 = h5file.create_group(sensor+'/B/PM_CF/PM10')
    b_cf_p10_data = h5file.create_group(sensor+'/B/PM_CF/PM10/Data')
    b_cf_p10_sub = h5file.create_group(sensor+'/B/PM_CF/PM10/Subsampled')
    b_pressure = h5file.create_group(sensor+'/B/Pressure')
    b_pressure_data = h5file.create_group(sensor+'/B/Pressure/Data')
    b_pressure_sub = h5file.create_group(sensor+'/B/Pressure/Subsampled')
    b_counts = h5file.create_group(sensor+'/B/Counts')
    b_c_03 = h5file.create_group(sensor+'/B/Counts/PM03')
    b_c_03_data = h5file.create_group(sensor+'/B/Counts/PM03/Data')
    b_c_03_sub = h5file.create_group(sensor+'/B/Counts/PM03/Subsampled')
    b_c_05 = h5file.create_group(sensor+'/B/Counts/PM05')
    b_c_05_data = h5file.create_group(sensor+'/B/Counts/PM05/Data')
    b_c_05_sub = h5file.create_group(sensor+'/B/Counts/PM05/Subsampled')
    b_c_1 = h5file.create_group(sensor+'/B/Counts/PM1')
    b_c_1_data = h5file.create_group(sensor+'/B/Counts/PM1/Data')
    b_c_1_sub = h5file.create_group(sensor+'/B/Counts/PM1/Subsampled')
    b_c_25 = h5file.create_group(sensor+'/B/Counts/PM25')
    b_c_25_data = h5file.create_group(sensor+'/B/Counts/PM25/Data')
    b_c_25_sub = h5file.create_group(sensor+'/B/Counts/PM25/Subsampled')
    b_c_10 = h5file.create_group(sensor+'/B/Counts/PM10')
    b_c_10_data = h5file.create_group(sensor+'/B/Counts/PM10/Data')
    b_c_10_sub = h5file.create_group(sensor+'/B/Counts/PM10/Subsampled')
    try:
        a_r_p1_data.create_dataset('PM1_Raw', data=dfpa['PM1_Raw_A'].values)
    except:
        print('Failed to fill: '+sensor)
    else:
        t_time_idx = []
        pa_time_idx = []
        sa_time_idx = []
        pb_time_idx = []
        sb_time_idx = []
        
        for i in range(0, len(dfsum.index.values)):
            t_time_idx.append(dfsum.index[i].to_julian_date())
        for i in range(0, len(dfpa['Time'].values)):
            pa_time_idx.append(pd.to_datetime(dfpa['Time'].iloc[i]).to_julian_date())
        for i in range(0, len(dfsa['Time'].values)):
            sa_time_idx.append(pd.to_datetime(dfsa['Time'].iloc[i]).to_julian_date())
        for i in range(0, len(dfpb['Time'].values)):
            pb_time_idx.append(pd.to_datetime(dfpb['Time'].iloc[i]).to_julian_date())
        for i in range(0, len(dfsb['Time'].values)):
            sb_time_idx.append(pd.to_datetime(dfsb['Time'].iloc[i]).to_julian_date())
        
        time.create_dataset('Subsampled_Time', data=t_time_idx)
        t_pa.create_dataset('Primary_Time', data=pa_time_idx)
        t_sa.create_dataset('Secondary_Time', data=sa_time_idx)
        t_pb.create_dataset('Primary_Time', data=pb_time_idx)
        t_sb.create_dataset('Secondary_Time', data=sb_time_idx)
        
        
        a_r_p1_sub.create_dataset('PM1_Raw_Mean', data=dfsum['PM1_Raw_A_mean'].values)
        a_r_p1_sub.create_dataset('PM1_Raw_Median', data=dfsum['PM1_Raw_A_med'].values)
        a_r_p1_sub.create_dataset('PM1_Raw_Std', data=dfsum['PM1_Raw_A_std'].values)
        a_r_p1_sub.create_dataset('PM1_Raw_Max', data=dfsum['PM1_Raw_A_max'].values)
        
        b_r_p1_data.create_dataset('PM1_Raw', data=dfpb['PM1_Raw_B'].values)
        b_r_p1_sub.create_dataset('PM1_Raw_Mean',data=dfsum['PM1_Raw_B_mean'].values)
        b_r_p1_sub.create_dataset('PM1_Raw_Median', data=dfsum['PM1_Raw_B_med'].values)
        b_r_p1_sub.create_dataset('PM1_Raw_Std', data=dfsum['PM1_Raw_B_std'].values)
        b_r_p1_sub.create_dataset('PM1_Raw_Max', data=dfsum['PM1_Raw_B_max'].values)

        a_cf_p1_data.create_dataset('PM1_CF', data=dfsa['PM1_CF_A'].values)
        a_cf_p1_sub.create_dataset('PM1_CF_Mean',data=dfsum['PM1_CF_A_mean'].values)
        a_cf_p1_sub.create_dataset('PM1_CF_Median', data=dfsum['PM1_CF_A_med'].values)
        a_cf_p1_sub.create_dataset('PM1_CF_Std', data=dfsum['PM1_CF_A_std'].values)
        a_cf_p1_sub.create_dataset('PM1_CF_Max',data=dfsum['PM1_CF_A_max'].values)

        b_cf_p1_data.create_dataset('PM1_CF', data=dfsb['PM1_CF_B'].values)
        b_cf_p1_sub.create_dataset('PM1_CF_Mean', data=dfsum['PM1_CF_B_mean'].values)
        b_cf_p1_sub.create_dataset('PM1_CF_Median', data=dfsum['PM1_CF_B_med'].values)
        b_cf_p1_sub.create_dataset('PM1_CF_Std', data=dfsum['PM1_CF_B_std'].values)
        b_cf_p1_sub.create_dataset('PM1_CF_Max', data=dfsum['PM1_CF_B_max'].values)
    
        a_r_p25_data.create_dataset('PM25_Raw', data=dfpa['PM25_Raw_A'].values)
        a_r_p25_sub.create_dataset('PM25_Raw_Mean', data=dfsum['PM25_Raw_A_mean'].values)
        a_r_p25_sub.create_dataset('PM25_Raw_Median', data=dfsum['PM25_Raw_A_med'].values)
        a_r_p25_sub.create_dataset('PM25_Raw_Std', data=dfsum['PM25_Raw_A_std'].values)
        a_r_p25_sub.create_dataset('PM25_Raw_Max', data=dfsum['PM25_Raw_A_max'].values)
        
        b_r_p25_data.create_dataset('PM25_Raw', data=dfpb['PM25_Raw_B'].values)
        b_r_p25_sub.create_dataset('PM25_Raw_Mean',data=dfsum['PM25_Raw_B_mean'].values)
        b_r_p25_sub.create_dataset('PM25_Raw_Median', data=dfsum['PM25_Raw_B_med'].values)
        b_r_p25_sub.create_dataset('PM25_Raw_Std', data=dfsum['PM25_Raw_B_std'].values)
        b_r_p25_sub.create_dataset('PM25_Raw_Max', data=dfsum['PM25_Raw_B_max'].values)

        a_cf_p25_data.create_dataset('PM25_CF', data=dfpa['PM25_CF_A'].values)
        a_cf_p25_sub.create_dataset('PM25_CF_Mean',data=dfsum['PM25_CF_A_mean'].values)
        a_cf_p25_sub.create_dataset('PM25_CF_Median', data=dfsum['PM25_CF_A_med'].values)
        a_cf_p25_sub.create_dataset('PM25_CF_Std', data=dfsum['PM25_CF_A_std'].values)
        a_cf_p25_sub.create_dataset('PM25_CF_Max',data=dfsum['PM25_CF_A_max'].values)

        b_cf_p25_data.create_dataset('PM25_CF', data=dfpb['PM25_CF_B'].values)
        b_cf_p25_sub.create_dataset('PM25_CF_Mean', data=dfsum['PM25_CF_B_mean'].values)
        b_cf_p25_sub.create_dataset('PM25_CF_Median', data=dfsum['PM25_CF_B_med'].values)
        b_cf_p25_sub.create_dataset('PM25_CF_Std', data=dfsum['PM25_CF_B_std'].values)
        b_cf_p25_sub.create_dataset('PM25_CF_Max', data=dfsum['PM25_CF_B_max'].values)
    
        a_r_p10_data.create_dataset('PM10_Raw', data=dfpa['PM10_Raw_A'].values)
        a_r_p10_sub.create_dataset('PM10_Raw_Mean', data=dfsum['PM10_Raw_A_mean'].values)
        a_r_p10_sub.create_dataset('PM10_Raw_Median', data=dfsum['PM10_Raw_A_med'].values)
        a_r_p10_sub.create_dataset('PM10_Raw_Std', data=dfsum['PM10_Raw_A_std'].values)
        a_r_p10_sub.create_dataset('PM10_Raw_Max', data=dfsum['PM10_Raw_A_max'].values)
        
        b_r_p10_data.create_dataset('PM10_Raw', data=dfpb['PM10_Raw_B'].values)
        b_r_p10_sub.create_dataset('PM10_Raw_Mean',data=dfsum['PM10_Raw_B_mean'].values)
        b_r_p10_sub.create_dataset('PM10_Raw_Median', data=dfsum['PM10_Raw_B_med'].values)
        b_r_p10_sub.create_dataset('PM10_Raw_Std', data=dfsum['PM10_Raw_B_std'].values)
        b_r_p10_sub.create_dataset('PM10_Raw_Max', data=dfsum['PM10_Raw_B_max'].values)

        a_cf_p10_data.create_dataset('PM10_CF', data=dfsa['PM10_CF_A'].values)
        a_cf_p10_sub.create_dataset('PM10_CF_Mean',data=dfsum['PM10_CF_A_mean'].values)
        a_cf_p10_sub.create_dataset('PM10_CF_Median', data=dfsum['PM10_CF_A_med'].values)
        a_cf_p10_sub.create_dataset('PM10_CF_Std', data=dfsum['PM10_CF_A_std'].values)
        a_cf_p10_sub.create_dataset('PM10_CF_Max',data=dfsum['PM10_CF_A_max'].values)

        b_cf_p10_data.create_dataset('PM10_CF', data=dfsb['PM10_CF_B'].values)
        b_cf_p10_sub.create_dataset('PM10_CF_Mean', data=dfsum['PM10_CF_B_mean'].values)
        b_cf_p10_sub.create_dataset('PM10_CF_Median', data=dfsum['PM10_CF_B_med'].values)
        b_cf_p10_sub.create_dataset('PM10_CF_Std', data=dfsum['PM10_CF_B_std'].values)
        b_cf_p10_sub.create_dataset('PM10_CF_Max', data=dfsum['PM10_CF_B_max'].values)
        
        a_temp_data.create_dataset('Raw_Temperature', data=dfpa['Temperature_A'].values)
        a_temp_sub.create_dataset('Temperature_Mean',data=dfsum['Temperature_A_mean'].values)
        a_temp_sub.create_dataset('Temperature_Median', data=dfsum['Temperature_A_med'].values)
        a_temp_sub.create_dataset('Temperature_Std', data=dfsum['Temperature_A_std'].values)
        a_temp_sub.create_dataset('Temperature_Max', data=dfsum['Temperature_A_max'].values)

        b_pressure_data.create_dataset('Raw_Pressure', data=dfpb['Pressure_B'].values)
        b_pressure_sub.create_dataset('Pressure_Mean',data=dfsum['Pressure_B_mean'].values)
        b_pressure_sub.create_dataset('Pressure_Median', data=dfsum['Pressure_B_med'].values)
        b_pressure_sub.create_dataset('Pressure_Std', data=dfsum['Pressure_B_std'].values)
        b_pressure_sub.create_dataset('Pressure_Max', data=dfsum['Pressure_B_max'].values)

        a_rh_data.create_dataset('Raw_RH', data=dfpa['RH_A'].values)
        a_rh_sub.create_dataset('RH_Mean',data=dfsum['RH_A_mean'].values)
        a_rh_sub.create_dataset('RH_Median', data=dfsum['RH_A_med'].values)
        a_rh_sub.create_dataset('RH_Std', data=dfsum['RH_A_std'].values)
        a_rh_sub.create_dataset('RH_Max', data=dfsum['RH_A_max'].values)
        
        a_c_03_data.create_dataset('Raw_PM03_dl', data=dfsa['PM03_dl_A'].values)
        a_c_03_sub.create_dataset('PM03_dl_Mean', data =dfsum['PM03_dl_A_mean'].values)
        a_c_03_sub.create_dataset('PM03_dl_Median', data =dfsum['PM03_dl_A_med'].values)
        a_c_03_sub.create_dataset('PM03_dl_Std', data =dfsum['PM03_dl_A_std'].values)
        a_c_03_sub.create_dataset('PM03_dl_Max', data =dfsum['PM03_dl_A_max'].values)
        
        a_c_05_data.create_dataset('Raw_PM05_dl', data=dfsa['PM05_dl_A'].values)
        a_c_05_sub.create_dataset('PM05_dl_Mean', data =dfsum['PM05_dl_A_mean'].values)
        a_c_05_sub.create_dataset('PM05_dl_Median', data =dfsum['PM05_dl_A_med'].values)
        a_c_05_sub.create_dataset('PM05_dl_Std', data =dfsum['PM05_dl_A_std'].values)
        a_c_05_sub.create_dataset('PM05_dl_Max', data =dfsum['PM05_dl_A_max'].values)
        
        a_c_1_data.create_dataset('Raw_PM1_dl', data=dfsa['PM1_dl_A'].values)
        a_c_1_sub.create_dataset('PM1_dl_Mean', data =dfsum['PM1_dl_A_mean'].values)
        a_c_1_sub.create_dataset('PM1_dl_Median', data =dfsum['PM1_dl_A_med'].values)
        a_c_1_sub.create_dataset('PM1_dl_Std', data =dfsum['PM1_dl_A_std'].values)
        a_c_1_sub.create_dataset('PM1_dl_Max', data =dfsum['PM1_dl_A_max'].values)
        
        a_c_25_data.create_dataset('Raw_PM25_dl', data=dfsa['PM25_dl_A'].values)
        a_c_25_sub.create_dataset('PM25_dl_Mean', data =dfsum['PM25_dl_A_mean'].values)
        a_c_25_sub.create_dataset('PM25_dl_Median', data =dfsum['PM25_dl_A_med'].values)
        a_c_25_sub.create_dataset('PM25_dl_Std', data =dfsum['PM25_dl_A_std'].values)
        a_c_25_sub.create_dataset('PM25_dl_Max', data =dfsum['PM25_dl_A_max'].values)
        
        a_c_10_data.create_dataset('Raw_PM10_dl', data=dfsa['PM10_dl_A'].values)
        a_c_10_sub.create_dataset('PM10_dl_Mean', data =dfsum['PM10_dl_A_mean'].values)
        a_c_10_sub.create_dataset('PM10_dl_Median', data =dfsum['PM10_dl_A_med'].values)
        a_c_10_sub.create_dataset('PM10_dl_Std', data =dfsum['PM10_dl_A_std'].values)
        a_c_10_sub.create_dataset('PM10_dl_Max', data =dfsum['PM10_dl_A_max'].values)
        
        b_c_03_data.create_dataset('Raw_PM03_dl', data=dfsb['PM03_dl_B'].values)
        b_c_03_sub.create_dataset('PM03_dl_Mean', data =dfsum['PM03_dl_B_mean'].values)
        b_c_03_sub.create_dataset('PM03_dl_Median', data =dfsum['PM03_dl_B_med'].values)
        b_c_03_sub.create_dataset('PM03_dl_Std', data =dfsum['PM03_dl_B_std'].values)
        b_c_03_sub.create_dataset('PM03_dl_Max', data =dfsum['PM03_dl_B_max'].values)
        
        b_c_05_data.create_dataset('Raw_PM05_dl', data=dfsb['PM05_dl_B'].values)
        b_c_05_sub.create_dataset('PM05_dl_Mean', data =dfsum['PM05_dl_B_mean'].values)
        b_c_05_sub.create_dataset('PM05_dl_Median', data =dfsum['PM05_dl_B_med'].values)
        b_c_05_sub.create_dataset('PM05_dl_Std', data =dfsum['PM05_dl_B_std'].values)
        b_c_05_sub.create_dataset('PM05_dl_Max', data =dfsum['PM05_dl_B_max'].values)
        
        b_c_1_data.create_dataset('Raw_PM1_dl', data=dfsb['PM1_dl_B'].values)
        b_c_1_sub.create_dataset('PM1_dl_Mean', data =dfsum['PM1_dl_B_mean'].values)
        b_c_1_sub.create_dataset('PM1_dl_Median', data =dfsum['PM1_dl_B_med'].values)
        b_c_1_sub.create_dataset('PM1_dl_Std', data =dfsum['PM1_dl_B_std'].values)
        b_c_1_sub.create_dataset('PM1_dl_Max', data =dfsum['PM1_dl_B_max'].values)
        
        b_c_25_data.create_dataset('Raw_PM25_dl', data=dfsb['PM25_dl_B'].values)
        b_c_25_sub.create_dataset('PM25_dl_Mean', data =dfsum['PM25_dl_B_mean'].values)
        b_c_25_sub.create_dataset('PM25_dl_Median', data =dfsum['PM25_dl_B_med'].values)
        b_c_25_sub.create_dataset('PM25_dl_Std', data =dfsum['PM25_dl_B_std'].values)
        b_c_25_sub.create_dataset('PM25_dl_Max', data =dfsum['PM25_dl_B_max'].values)
        
        b_c_10_data.create_dataset('Raw_PM10_dl', data=dfsb['PM10_dl_B'].values)
        b_c_10_sub.create_dataset('PM10_dl_Mean', data =dfsum['PM10_dl_B_mean'].values)
        b_c_10_sub.create_dataset('PM10_dl_Median', data =dfsum['PM10_dl_B_med'].values)
        b_c_10_sub.create_dataset('PM10_dl_Std', data =dfsum['PM10_dl_B_std'].values)
        b_c_10_sub.create_dataset('PM10_dl_Max', data =dfsum['PM10_dl_B_max'].values)
    h5file.flush()
    return h5file

def build_hdf(name_list, hdfname, tzstr, dir_name):
    '''
    Scripts fill_hdf function based on list of unique sensor names.
    Closes hdfile object.
    '''
    h5file= file_hdf(hdfname)
    sensors = []
    for i in range(0, len(name_list)):
        try:
            pa = name_list[i][0]
            sa = name_list[i][1]
        except:
            print('no data from IDX: '+str(i))
        else:
            try:
                pb = name_list[i][2]
                sb = name_list[i][3]
            except:
                pb = pa
                sb = sa
                no_b = True
            else:
                no_b = False
            sensors.append(name_list[i][0].replace('Primary_','').split('_20')[0])
            print(i,sensors[i])
            pa = pd.read_csv(Path(dir_name+'/'+pa), skip_blank_lines= False)
            sa = pd.read_csv(Path(dir_name+'/'+sa), skip_blank_lines= False)
            pb = pd.read_csv(Path(dir_name+'/'+pb), skip_blank_lines= False)
            sb = pd.read_csv(Path(dir_name+'/'+sb), skip_blank_lines= False)
            #pb.iloc[1,-1] = 1776
            pa.dropna(axis=1, how='all',inplace=True)
            pb.dropna(axis=1, how='all',inplace=True)
            sa.dropna(axis=1, how='all',inplace=True)
            sb.dropna(axis=1, how='all',inplace=True)
            if (len(pa.columns) == 10):
                lpa = ['Time','entry_id','PM1_Raw_A', 'PM25_Raw_A','PM10_Raw_A',
                              'Uptime','ADC','Temperature_A','RH_A','PM25_CF_A']
                lpb = ['Time','entry_id','PM1_Raw_B', 'PM25_Raw_B','PM10_Raw_B',
                              'Uptime','ADC','Pressure_B','__','PM25_CF_B']
                lsa = ['Time','entry_id','PM03_dl_A','PM05_dl_A','PM1_dl_A',
                             'PM25_dl_A','PM5_dl_A','PM10_dl_A','PM1_CF_A','PM10_CF_A']
                lsb = ['Time','entry_id','PM03_dl_B','PM05_dl_B','PM1_dl_B',
                             'PM25_dl_B','PM5_dl_B','PM10_dl_B','PM1_CF_B','PM10_CF_B']
                dpa = ['entry_id','Uptime','ADC']
                dpb = ['entry_id','Uptime','ADC','__']
                dsa = ['entry_id']
                dsb = ['entry_id']
            else:
                lpa = ['Time','PM1_Raw_A', 'PM25_Raw_A','PM10_Raw_A',
                              'Uptime','ADC','Temperature_A','RH_A','PM25_CF_A']
                lpb = ['Time','PM1_Raw_B', 'PM25_Raw_B','PM10_Raw_B',
                              'Uptime','ADC','Pressure_B','__','PM25_CF_B']
                lsa = ['Time','PM03_dl_A','PM05_dl_A','PM1_dl_A',
                             'PM25_dl_A','PM5_dl_A','PM10_dl_A','PM1_CF_A','PM10_CF_A']
                lsb = ['Time','PM03_dl_B','PM05_dl_B','PM1_dl_B',
                             'PM25_dl_B','PM5_dl_B','PM10_dl_B','PM1_CF_B','PM10_CF_B']
                dpa = ['Uptime','ADC']
                dpb = ['Uptime','ADC','__']
                dsa = []
                dsb = []
            try:
                len(pa.iloc[:,2]) != 0
                len(pb.iloc[:,2]) != 0
                if (pa.shape[0] > 0) :
                    pa.columns = lpa
                    sa.columns = lsa
                    pb.columns = lpb
                    sb.columns = lsb
                    pa.drop(dpa, inplace=True, axis=1)
                    sa.drop(dsa, inplace=True, axis=1)
                    pb.drop(dpb, inplace=True, axis=1)
                    sb.drop(dsb, inplace=True, axis=1)
                else:
                    """
                    pa.columns = ['Time','entry_id','PM1_Raw_A', 'PM25_Raw_A','PM10_Raw_A',
                                  'Uptime','ADC','Temperature_A','RH_A','PM25_CF_A']
                    sa.columns = ['Time','entry_id','PM03_dl_A','PM05_dl_A','PM1_dl_A',
                                 'PM25_dl_A','PM5_dl_A','PM10_dl_A','PM1_CF_A','PM10_CF_A']
                    pb.columns = ['Time','entry_id','PM1_Raw_B', 'PM25_Raw_B','PM10_Raw_B',
                                  'Uptime','ADC','Pressure_B','__','PM25_CF_B']
                    sb.columns = ['Time','entry_id','PM03_dl_B','PM05_dl_B','PM1_dl_B',
                                 'PM25_dl_B','PM5_dl_B','PM10_dl_B','PM1_CF_B','PM10_CF_B']
                    pa.drop(['entry_id','Uptime','ADC'], inplace=True, axis=1)
                    sa.drop(['entry_id'],inplace=True, axis = 1)
                    pb.drop(['entry_id''Uptime','ADC','__'], inplace=True, axis=1)
                    sb.drop(['entry_id'],inplace=True, axis = 1)
                    """
                    na_fill = []
                    for kj in range(0, len(lpa)):
                        na_fill.append(np.nan)
                    pa = pd.DataFrame(na_fill).T
                    sa = pd.DataFrame(na_fill).T
                    pb = pd.DataFrame(na_fill).T
                    sb = pd.DataFrame(na_fill).T
                    pa.columns = lpa
                    sa.columns = lsa
                    pb.columns = lpb
                    sb.columns = lsb
                    pa.drop(dpa, inplace=True, axis=1)
                    sa.drop(dsa, inplace=True, axis=1)
                    pb.drop(dpb, inplace=True, axis=1)
                    sb.drop(dsb, inplace=True, axis=1)

                    pa.iloc[0,0] = '1996-10-17 19:00:00 UTC'
                    pb.iloc[0,0] = '1996-10-17 19:00:00 UTC'
                    sa.iloc[0,0] = '1996-10-17 19:00:00 UTC'
                    sb.iloc[0,0] = '1996-10-17 19:00:00 UTC'
                if no_b == True:
                    pb.iloc[:,1:] = np.nan
                    sb.iloc[:,1:] = np.nan 

                df_summary_pa = time_master(pa, pa.columns, tzstr)
                df_summary_sa = time_master(sa, sa.columns, tzstr)
                df_summary_pb = time_master(pb, pb.columns, tzstr)
                df_summary_sb = time_master(sb, sb.columns, tzstr)

                df_summary = pd.concat([df_summary_pa, df_summary_sa, df_summary_pb, df_summary_sb], axis=1)
                h5file = fill_hdf(h5file, sensors[i], df_summary, pa, sa, pb, sb)
            except:
                print("No PM data stored.")
    h5file.close()

def load_dict_from_hdf5(filename):
    '''
    Builds python native hierchiacal format dictonary from hdf5 file.
    '''
    with h5.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    '''
    Recursively emulates file storage structure from hdf5 file to dictionary.
    '''
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def hdf5_to_mat(hdfile):
    '''
    Converts hdf5 file to MATLAB & scipy.io.loadmat compatible .mat file.
    '''
    data = load_dict_from_hdf5(hdfile)
    arr = np.array(list(data.items()))
    savemat(hdfile[:-3] + '.mat', {'arr':arr})
    print('Successfully built .mat file.')

def sensors_from_csv(csvfile):
    '''
    Converts sensor list from csvfile to list structure in python.
    Sensor names should be exact, and in the 1st column of the csv.
    '''
    sensors = pd.read_csv(csvfile)
    sensor_list = sensors.iloc[:,0].values.tolist()
    return sensor_list

def downloaded_file_list(directory, sensor_list):
    '''
    Function that arranges the names of the downloaded PA
    csv's into a nested list of grouped sensors.
    '''
    name_list = []
    for i in range(0, len(sensor_list)):
        search_name = directory +'*_'+sensor_list[i].replace(' ','_')+'_*.csv'
        name_list_temp = glob.glob(search_name)
        name_list = name_list + name_list_temp
    name_list = np.array(sorted(name_list))
    name_list = np.unique(name_list)
    name_list = name_list.tolist()
    name_idx = name_list.copy()
    name_idx_primary = []
    name_idx_secondary = []
    for i in range(0, len(name_idx)):
        name_temp = name_idx[i]
        name_temp = name_temp.split('/')[-1]
        name_temp = name_temp.replace('_B_','_')
        if (name_temp.find('Primary') != -1):
            name_idx_primary.append(name_temp)
        else:
            name_idx_secondary.append(name_temp)
    _,ri,rc = np.unique(name_idx_primary, return_index=True, return_counts=True)
    names = []
    for i in range(0, len(ri)):
        s = ri[i]
        e = ri[i]+rc[i]
        names_to_append = [name_idx_primary[s:e], name_idx_secondary[s:e]]
        flat_append = []
        for sublist in names_to_append:
            for item in sublist:
                flat_append.append(item)
        names.append(flat_append)
    return names

def h5file_query(h5file, query_string):
    f = h5.File(h5file, 'r')
    array = f[query_string][:]
    return array

def a_vs_b_plot(h5file, sensor, show):
    '''
    Generates intra-sensor correlation plots.
    Very important for understanding the relability
    of sensor data. Requires the h5file, the sensor
    name, and the show. Returns the important 
    correlation statistics, and saves as a png.
    '''
    sname = sensor+'_AB_plot.png'
    A_PM25_hr = h5file_query(h5file, sensor + ' /A/PM_CF/PM25/Subsampled/PM25_CF_Mean')
    B_PM25_hr = h5file_query(h5file, sensor + ' /B/PM_CF/PM25/Subsampled/PM25_CF_Mean')
    mask = ~np.isnan(A_PM25_hr) & ~np.isnan(B_PM25_hr)
    axlim = np.percentile(A_PM25_hr[mask]+B_PM25_hr[mask],95)
    axlim = int(math.ceil(axlim / 10.0)) * 10
    m, b, R,_,_ = stats.linregress(A_PM25_hr[mask], B_PM25_hr[mask])
    B_fitted = A_PM25_hr*m+b
    r,_ = stats.pearsonr(A_PM25_hr[mask], B_PM25_hr[mask])
    rmse = metrics.mean_squared_error(B_PM25_hr[mask], B_fitted[mask])
    nrmse = rmse/np.nanmean(B_PM25_hr)
    OLS = 'B = '+str(np.round(m,2))+' * A + '+str(np.round(b,2))
    fig,ax = plt.subplots(figsize=(8,8))
    plt.plot(A_PM25_hr,B_fitted,c='purple')
    plt.scatter(A_PM25_hr, B_PM25_hr,c='purple',s=100)
    plt.plot(A_PM25_hr,A_PM25_hr,c='k')
    plt.grid(alpha=0.35)
    plt.xlabel('Sensor A PM$_{2.5}$ ($\mu$g/m$^3$)',fontname='Arial', fontsize=20)
    plt.ylabel('Sensor B PM$_{2.5}$ ($\mu$g/m$^3$)',fontname='Arial', fontsize=20)
    plt.xlim([0, axlim])
    plt.ylim([0, axlim])
    plt.title(sensor, fontname='Arial', fontsize=24)
    plt.rcParams.update({'font.size': 16, 'font.sans-serif':['Arial']})
    plt.legend(['1-hr Average Measurement','1:1 Line',OLS], 
               framealpha=1, edgecolor='k',loc=2)
    plt.text(.68*axlim,.15*axlim,'OLS R$^2$       : '+ str(np.round(R**2,3)))
    plt.text(.68*axlim,.10*axlim,"Pearson's r : "+str(np.round(r,3)))
    plt.text(.68*axlim,.05*axlim, "NRMSE      : "+str(np.round(nrmse,3)))
    plt.savefig(sname)
    if show==True:
        plt.show()
    return m, b, R, r, nrmse

def calibration_plot(h5file, sensor, calibration_df,show):
    '''
    Generates calibration correlation plots.
    Very important for understanding the relability
    of sensor data. Requires the h5file, the sensor
    name, a calibration dataframe, and the show. 
    Returns the important correlation statistics, 
    and saves as a png.
    '''
    sname = sensor+'_Calibration_plot.png'
    A_PM25_hr = h5file_query(h5file, sensor + ' /A/PM_CF/PM25/Subsampled/PM25_CF_Mean')
    B_PM25_hr = h5file_query(h5file, sensor + ' /B/PM_CF/PM25/Subsampled/PM25_CF_Mean')
    julian_time = h5file_query(h5file, sensor + ' /Time/Subsampled_Time')
    time = []
    for i in range(0, len(julian_time)):
        t = jd.jd_to_datetime(julian_time[i])
        t2 = pd.Timestamp(datetime(t.year,t.month,t.day,t.hour,0,0))
        time = time + [t2]
    PA = pd.DataFrame([A_PM25_hr,B_PM25_hr])
    PA = PA.T
    PA.index = time
    PA_Calibration = pd.concat([PA, calibration_df])
    PA_Calibration.columns = [sensor+'A',sensor+'B','Calibration']
    PA_Calibration = PA_Calibration.resample('60T').apply(np.nanmean)
    mask = ~np.isnan(
        PA_Calibration.iloc[:,0].values) & ~np.isnan(
        PA_Calibration.iloc[:,1].values) & ~np.isnan(
        PA_Calibration.iloc[:,2].values)

    A = PA_Calibration.iloc[:,0].values[mask]
    B = PA_Calibration.iloc[:,1].values[mask]
    C = PA_Calibration.iloc[:,2].values[mask]
    I = PA_Calibration.index[mask]
    ma, ba, Ra,_,_ = stats.linregress(C,A)
    mb, bb, Rb,_,_ = stats.linregress(C,B)
    OLSA = 'PA = '+str(np.round(ma,3))+' * Calibration + ' + str(np.round(ba,3))
    OLSB = 'PA = '+str(np.round(mb,3))+' * Calibration + ' + str(np.round(bb,3))
    A_fitted = C*ma+ba
    B_fitted = C*mb+bb
    axlim = np.percentile(A+C,95)
    axlim = int(math.ceil(axlim / 10.0)) * 10
    ra,_ = stats.pearsonr(A, C)
    rmsea = metrics.mean_squared_error(A, A_fitted)
    nrmsea = rmse/np.nanmean(A)
    rb,_ = stats.pearsonr(B, C)
    rmseb = metrics.mean_squared_error(B, B_fitted)
    nrmseb = rmse/np.nanmean(B)
    fig,ax = plt.subplots(1,2,figsize=(17,8))
    ax[0].scatter(A,C,c='purple')
    ax[0].plot(C, A_fitted, c='purple')
    ax[0].plot(C, C, c='k')
    ax[0].set_xlim([0,axlim])
    ax[0].set_ylim([0,axlim])
    ax[0].grid(alpha=0.35)
    ax[0].set_xlabel('Calibration PM$_{2.5}$ ($\mu$g/m$^3$)',fontname='Arial', fontsize=20)
    ax[0].set_ylabel('PA PM$_{2.5}$ ($\mu$g/m$^3$)',fontname='Arial', fontsize=20)
    plt.rcParams.update({'font.size': 16, 'font.sans-serif':['Arial']})
    ax[0].legend(['1-hr Average Measurement','1:1 Line',OLSA], 
                   framealpha=1, edgecolor='k',loc=2)
    ax[0].text(.68*axlim,.15*axlim,'OLS R$^2$       : '+ str(np.round(Ra**2,3)))
    ax[0].text(.68*axlim,.10*axlim,"Pearson's r : "+str(np.round(ra,3)))
    ax[0].text(.68*axlim,.05*axlim, "NRMSE      : "+str(np.round(nrmsea,3)))
    ax[0].set_title(sensor+'A OLS Calibration')
    ax[1].scatter(B,C,c='purple')
    ax[1].plot(C, B_fitted, c='purple')
    ax[1].plot(C, C, c='k')
    ax[1].set_xlim([0,axlim])
    ax[1].set_ylim([0,axlim])
    ax[1].grid(alpha=0.35)
    ax[1].set_xlabel('Calibration PM$_{2.5}$ ($\mu$g/m$^3$)',fontname='Arial', fontsize=20)
    ax[1].set_ylabel('PA PM$_{2.5}$ ($\mu$g/m$^3$)',fontname='Arial', fontsize=20)
    plt.rcParams.update({'font.size': 16, 'font.sans-serif':['Arial']})
    ax[1].legend(['1-hr Average Measurement','1:1 Line',OLSB], 
                   framealpha=1, edgecolor='k',loc=2)
    ax[1].text(.68*axlim,.15*axlim,'OLS R$^2$       : '+ str(np.round(Rb**2,3)))
    ax[1].text(.68*axlim,.10*axlim,"Pearson's r : "+str(np.round(rb,3)))
    ax[1].text(.68*axlim,.05*axlim, "NRMSE      : "+str(np.round(nrmseb,3)))
    ax[1].set_title(sensor+'B OLS Calibration')
    plt.savefig(sname)
    if show==True:
        plt.show()
    a_stats = [ma, ba, Ra, ra, nrmsea]
    b_stats = [mb, bb, Rb, rb, nrmseb]
    return a_stats, b_stats

def map_df(h5file, sensor_list):
    placeholder = pd.Timestamp(datetime(1996,10,17,7,0,0))
    mp_df = pd.DataFrame([np.nan], index=[placeholder])
    sl = []
    ab = 0
    for i in range(0, len(sensor_list)):
        time = []
        PM25 = []
        sensor = sensor_list[i]
        print(sensor)
        try:
            A_PM25 = h5file_query(h5file, sensor+'/A/PM_CF/PM25/Subsampled/PM25_CF_Mean')
            B_PM25 = h5file_query(h5file, sensor+ '/B/PM_CF/PM25/Subsampled/PM25_CF_Mean')
            julian_time = h5file_query(h5file, sensor + '/Time/Subsampled_Time')
            for j in range(0, len(julian_time)):
                t = jd.jd_to_datetime(julian_time[j])
                if t.minute >= 30:
                    t2 = pd.Timestamp(datetime(t.year,t.month,t.day,t.hour+1,0,0))
                else:
                    t2 = pd.Timestamp(datetime(t.year,t.month,t.day,t.hour,0,0))
                time.append(t2)
                pm25 = A_PM25[j]
                '''ERROR = np.abs(A_PM25[j] - B_PM25[j])/A_PM25[j]*100
                if ERROR >= 15:
                    ab = ab+1
                    print(ab)'''
                PM25.append(pm25)
            temp_df = pd.DataFrame([PM25])
            temp_df = temp_df.T
            temp_df.index = time
            temp_df = temp_df.resample('60T').apply(np.mean)
            mp_df = pd.concat([mp_df, temp_df], axis=1)
            sl.append(sensor)
        except:
            print('')
    mp_df.columns = ['placer']+sl
    mp_df = mp_df.drop(['placer'], axis=1)
    mp_df = mp_df.drop([placeholder])
    return mp_df
        
def make_marker(x,y,c,m,sensor):
    cmap = mpl.cm.get_cmap('hot_r')
    rgba = col.to_hex(cmap(c/65))
    folium.CircleMarker(
        radius=10,
        location=[x, y],
        popup=sensor+'\n Concentration: '+str(np.round(c,2))+' ug/m3',
        color=rgba,
        fill=True,
        fillcolor=rgba,
        fill_opacity=1.0
    ).add_to(m)
    return m
    
def make_map(x,y,c,latlon,sname):
    m = folium.Map(location=[x, y],
               tiles='https://tiles.wmflabs.org/osm-no-labels/{z}/{x}/{y}.png ',
               attr='Wikimedia')
    for i in range(0,len(latlon)):
        sensor = latlon.Sensor[i]
        xs = latlon.Latitude[i]
        ys = latlon.Longitude[i]
        conc = c[i]
        make_marker(xs,ys,conc,m, sensor)
    m.save(sname)
    return m

def save_map(html_list, bulk, driver_path):
    driver = webdriver.Chrome(driver_path)
    if bulk == False:
        driver.get('file://'+html_list)
        sleep(2.5)
        driver.save_screenshot(html_list.replace('html','png'))
    else:
        for i in range(0, len(html_list)):
            html = html_list[i]
            driver.get('file://'+html_list)
            sleep(2.5)
            driver.save_screenshot(html.replace('html','png'))
    driver.close()