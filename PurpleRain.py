from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
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
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import calendar as cal
import numpy as np
import pandas as pd
import h5py as h5
import os
import math
import glob
import warnings
import itertools
import jdutil as jd

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
	'''
	Determines if the input string
	can be converted to float.
	Takes the string as input, and
	returns false for any error.
	'''
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
    os.makedirs(dir_path)
    print('New folder created!')
    return dir_path

def server_call(driver_path):
    '''
    Function to access PurpleAir Sensor Server website via 
    Chrome with specficied preferences - need driver path.
    Driver can be dowloaded from: https://chromedriver.chromium.org/. 
    Returns driver to manipulate website.
    '''
    chrome_opts = Options()
    chrome_opts.add_argument(" - incognito")
    prefs = {'profile.default_content_setting_values.automatic_downloads': 1}
    chrome_opts.add_experimental_option("prefs", prefs)
    try:
        driver = webdriver.Chrome(driver_path, options=chrome_opts)
    except:
        print('Make sure you are connected to the internet.')
        print(
        'You must download the correct Chrome driver from: https://chromedriver.chromium.org/.')
        print('Make sure the Chrome driver you download matches your version of Chrome.')
        driver = np.nan
    else:
        driver.get('http://www.purpleair.com/sensorlist')
        print('Accessed PA server!')
    return driver

def download_selection(driver, IDpl, sd, ed):
    '''
    Takes driver, [A] Sensor [Primary] Serial Number, 
    Start Date, and End Date. Fills in time bounds, 
    checks download boxes and downloads 1 sensor at a
    time to avoid time truncation. Returns nothing.
    '''
    start_date = driver.find_element_by_xpath("//input[@id='startdatepicker']")
    ActionChains(driver).move_to_element(start_date).click().send_keys(sd).perform()
    end_date = driver.find_element_by_xpath("//input[@id='enddatepicker']")
    ActionChains(driver).move_to_element(end_date).click().send_keys(ed).perform()
    for i in range(0, len(IDpl)):
        IDp = IDpl[i]
        IDs = IDp+1
        try:
            box_path = "//input[@value='"+str(IDp)+'|'+str(IDs)+"']"
            box = driver.find_element_by_xpath(box_path)
        except:
            print('Unable to find sensor [index] : '+str(i))
        else:
            driver.execute_script("arguments[0].scrollIntoView();", box)
            box.click()
            try:
                box_path_B = "//input[@value='"+str(IDp+2)+'|'+str(IDs+2)+"']" 
                box_B = driver.find_element_by_xpath(box_path_B)
            except:
                box_path_B = "//input[@value='"+str(IDp+2)+'|'+str(IDs+3)+"']" 
                box_B = driver.find_element_by_xpath(box_path_B)
            else:
                box_B.click()
            download = driver.find_element_by_xpath("//button[@onclick='downloadall();']")
            download.click()
            unselect = driver.find_element_by_xpath("//button[@onclick='selectnone();']")
            unselect.click()
    print('Download complete!')
    
def build_database(driver_path):
    '''
    Function builds database of ALL PurpleAir sensors in 
    the entire world based off the website at the time
    of access. Purpose of the function is to record all 
    Lat-Lon info & Serial Numbers. This function
    is generally not needed to download data. If many of 
    your sensors have been downgraded, then it may be 
    necessary to use since Serial Numbers will not follow 
    the typical pattern: [Primary A], [Secondary A], [Primary B]
    [Secondary B].
    '''
    try:
        driver = server_call(driver_path)
        a = driver.find_elements_by_tag_name("a")
        b = driver.find_elements_by_tag_name("b")
        button = driver.find_elements_by_tag_name('button')
        button = [v for i, v in enumerate(button) if i % 2 == 0]
        del button[0]
        del a[0]
        del b[0]
        del b[0]
        del b[0]
        del b[len(b)-1]
        Id = []
        Lat = []
        Lon = []
        UnM = []
        UnT = []
        print('Scraped HTML/CSS/JS.')
        for i in range(0, len(a)):
            l = a[i].get_attribute("href").replace('http://www.purpleair.com/map?',
                                                   '').replace('zoom=12&','').split('&')
            if len(l) < 3 and len(l) > 0:
                Lat.append(np.nan)
                Lon.append(np.nan)
                UnT.append(int(l[0].replace('inc=','')))
            else:
                Lat.append(is_numf(l[0].replace('lat=','')))
                Lon.append(is_numf(l[1].replace('lng=','')))
                UnT.append(is_numi(l[2].replace('inc=','')))
        for i in range(0, len(b)):
            Id.append(str(b[i].text))
            UnM.append(int(button[i].get_attribute('id').replace('_download_button','')))
        Master = pd.DataFrame(Id)
        Master['Unique'] = UnM
        Master.columns = ['Id','Serial_No']
        LatLon = pd.DataFrame(Lat)
        LatLon['Lon'] = Lon
        LatLon['Unique'] = UnT
        LatLon.columns = ['Lat','Lon','Serial_No']
        MasterLatLon = pd.merge(Master,LatLon)
        MasterLatLon.to_csv('PurpleRainMasterLatLon.csv')
        Master.to_csv('PurpleRainMaster.csv')
        print('Built database csv files.')
    except:
        answer = 'Browser not availible make sure: 1) There is an internet connection & 2)         The chrome driver for this version of chrome is in the same folder as this script.'
    else:
        driver.close()
        answer = 'Database successfully built!'
    return answer

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

def file_hdf(name_list, sname):
    '''
    Intializes hdf file, and builds unique sensor list. 
    Returns writtable hdf file and unique sensor list.
    '''
    h5file = h5.File(sname+'.h5','w')
    for i in range(0, len(name_list)):
        sensor = name_list[i]
    return h5file, sensor_list

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

def build_hdf(name_list, hdfname, tzstr):
    '''
    Scripts fill_hdf function based on list of unique sensor names.
    Closes hdfile object.
    '''
    sensor_list = []
    h5file, sensor_list = file_hdf(np.unique(sensor_list), hdfname)
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
            print(pa)
            pa = pd.read_csv(pa)
            sa = pd.read_csv(sa)
            pb = pd.read_csv(pb)
            sb = pd.read_csv(sb)
            pb.iloc[1,8] = 1776
            pa.dropna(axis=1, how='all',inplace=True)
            pb.dropna(axis=1, how='all',inplace=True)
            sa.dropna(axis=1, how='all',inplace=True)
            sb.dropna(axis=1, how='all',inplace=True)
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
            if pa.shape[0] > 0:
                pa.columns = lpa
                sa.columns = lsa
                pb.columns = lpb
                sb.columns = lsb
                pa.drop(dpa, inplace=True, axis=1)
                sa.drop(dsa, inplace=True, axis=1)
                pb.drop(dpb, inplace=True, axis=1)
                sb.drop(dsb, inplace=True, axis=1)
            else:
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
                
                pa = pd.DataFrame(np.nan, columns = pa.columns, index = np.array([0]))
                sa = pd.DataFrame(np.nan, columns = sa.columns, index = np.array([0]))
                pb = pd.DataFrame(np.nan, columns = pb.columns, index = np.array([0]))
                sb = pd.DataFrame(np.nan, columns = sb.columns, index = np.array([0]))
                
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

            df_summary = pd.concat(
            [df_summary_pa, df_summary_sa, df_summary_pb, df_summary_sb], axis=1)
            sensor_list = name_list[i][0].replace(
            'Primary','').replace(' B ','').split('(')[0].split('/')[4]
            h5file = fill_hdf(h5file, sensor_list, df_summary, pa, sa, pb, sb)
    keys = h5file.keys()
    h5file.close()
    return keys

def pa_query(qtype, driver_path, qnames, sd, ed, tzstr, *n):
    '''
    The main function users interact with to call the server, 
    download the data, and organize into the hdf file.
    Takes querey type [0]: Exact Name given in sensor list, 
    [1]: Partial Name given in sensor list, [2]: Lat-lon bounds
    Takes driver_path for chrome, sensors to be used to query 
    [A] and [B] are considered on sensor.
    Start Date and end date as strings of the form: 'MM/DD/YYYY', 
    and timezone string of the form: 'Asia/Calcutta'.
    More info on timezones can be found at 
    http://pytz.sourceforge.net/#country-information.
    '''
    if qtype == 'Exact' or qtype == 'exact' or qtype == 0:
        driver = server_call(driver_path)
        IDp = []
        print("Download in process.")
        for i in range(0, len(qnames)):
            print(qnames[i])
            xpath = "//b[text()='"+str(qnames[i])+"'"+"]/following::a"
            selection = driver.find_element_by_xpath(xpath)
            IDp.append(int(selection.get_attribute(
            'href').replace('http://www.purpleair.com/map?lat=',
                                                             '').split(
                                                             '&')[3].replace('inc=','')))
        download_selection(driver, IDp, sd, ed)
    else:
        if len(n) == 1:
            if qtype == 'Inexact' or qtype =='inexact' or qtype == 1:
                master = pd.read_csv(n[0])
                for i in range(0, len(qnames)):
                    selection = qnames[i]
                    ind = np.where(selection in master.Id.values == True)
                    ind = ind[0].tolist()
                    IDp = master.Serial_No.values[ind]
                    download_selection(driver, IDp, sd, ed)
            elif qtype == 'LatLon' or qtype == 'latlon' or qtype == 'lat-lon' or qtype == 'Lat-Lon' or qtype == 2:
                master_latlon = pd.read_csv(n[0])
                ind = []
                for i in range(0,len(master_latlon.Serial_No)):
                    if master_latlon.Lat[i] >= n[1][0] and master_latlon.Lat[i] <= n[1][1]                    and master_latlon.Lon[i] >= n[1][2] and master_laton.Lon[i] <= n[1][3]:
                        ind.append(i)
                IDp = master_latlon.Serial_No.values[ind]
                download_selection(driver, IDp, sd, ed)
            else:
                    print('Unacceptable query type. The only acceptable queries are')
                    print('0 - Exact, 1 - Inexact, or 2 - LatLon.')
        elif len(n) > 1:
            print('Too many arguments for database path - only one string is needed.')
        else:
            print(
            'Database path has not been included - it is necessary for query types 1 & 2.')
    driver.close()
    
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
    Function that arranges the names ofthe downloaded PA
    csv's into a nested list of grouped sensors.
    '''
    name_list = []
    for i in range(0, len(sensor_list)):
        search_name = directory + sensor_list[i]+'*.csv'
        name_list_temp = glob.glob(search_name)
        name_list = name_list + name_list_temp
    name_list = np.array(sorted(name_list))
    name_list = np.unique(name_list)
    name_list = name_list.tolist()
    name_idx = name_list.copy()
    for i in range(0, len(name_idx)):
        name_temp = name_idx[i].split('(')[0].split('/')
        name_temp = name_temp[len(name_temp)-1].strip().replace(' B','').strip()
        name_idx[i] = name_temp
    _,ri,rc = np.unique(name_idx, return_index=True, return_counts=True)
    names = []
    for i in range(0, len(ri)):
        s = ri[i]
        e = ri[i]+rc[i]
        names.append(name_list[s:e])
    return names

def download_met_data(driver_path, network, stations, start_date, end_date):
    '''
    Function that automates downloads from the Mesonet - includes all ASOS
    (airport + special meterological sites). Requires driver path to
    chromedriver, network name (usually country name, or state in
    the USA), station (airport code in the USA), start date, and 
    end date as datetime structure, ie: datetime(year, month, day)
    '''
    start_yr = str(start_date.year)
    start_mn = cal.month_name[start_date.month]
    start_dy = str(start_date.day)
    end_yr = end_date.year
    end_mn = cal.month_name[end_date.month]
    end_dy = end_date.day
    chrome_opts = Options()
    chrome_opts.add_argument(" - incognito")
    prefs = {'profile.default_content_setting_values.automatic_downloads': 1}
    chrome_opts.add_experimental_option("prefs", prefs)
    try:
        driver = webdriver.Chrome(driver_path, options=chrome_opts)
    except:
        print('Make sure you are connected to the internet.')
        print(
        'You must download the correct Chrome driver from: https://chromedriver.chromium.org/.')
        print('Make sure the Chrome driver you download matches your version of Chrome.')
        driver = np.nan
    else:
        driver.get('https://mesonet.agron.iastate.edu/request/download.phtml')
        print('Accessed Mesonet server!')
    opts = driver.find_elements_by_tag_name("option")
    network_strings = []
    for i in range(0, len(opts)):
        network_strings_temp = [opts[i].text.replace('ASOS','').strip()]
        network_strings = network_strings+network_strings_temp
    network_matching = [s for s in network_strings if network == s]
    network_select = driver.find_element_by_xpath(
    "//*[contains(text(), '"+network_matching[0]+' ASOS'+"')]")
    network_code = network_select.get_attribute('value')
    driver.get(
    'https://mesonet.agron.iastate.edu/request/download.phtml?network='+network_code)
    station_entry = driver.find_element_by_xpath("//input[@id='stationfilter']")
    start_year = driver.find_element_by_xpath("//select[@name='year1']")
    start_month = driver.find_element_by_xpath("//select[@name='month1']")
    start_day = driver.find_element_by_xpath("//select[@name='day1']")
    end_year = driver.find_element_by_xpath("//select[@name='year2']")
    end_month = driver.find_element_by_xpath("//select[@name='month2']")
    end_day = driver.find_element_by_xpath("//select[@name='day2']")
    download_file = Select(driver.find_element_by_xpath("//select[@name='direct']"))
    get_data = driver.find_element_by_xpath("//input[@value='Get Data']")
    ActionChains(driver).move_to_element(
    station_entry).click().send_keys(stations).click().perform()
    ActionChains(driver).move_to_element(
    start_year).click().send_keys(start_yr).send_keys(Keys.ENTER).perform()
    ActionChains(driver).move_to_element(
    start_month).click().send_keys(start_mn).send_keys(Keys.ENTER).perform()
    ActionChains(driver).move_to_element(
    start_day).click().send_keys(start_dy).send_keys(Keys.ENTER).perform()
    ActionChains(driver).move_to_element(
    end_year).click().send_keys(end_yr).send_keys(Keys.ENTER).perform()
    ActionChains(driver).move_to_element(
    end_month).click().send_keys(end_mn).send_keys(Keys.ENTER).perform()
    ActionChains(driver).move_to_element(
    end_day).click().send_keys(end_dy).send_keys(Keys.ENTER).perform()
    download_file.select_by_visible_text('Save result data to file on computer')
    ActionChains(driver).move_to_element(get_data).click().perform()
    print(
    'Beginning Download - waiting 30 seconds for file to finish before closing browser.')
    sleep(30)
    driver.close()
    print('Downloaded File!')

def format_met_data(txtfile, tz_str):
    '''
    Function that formats and saves met data
    downloaded by function download_met_data.
    Takes the textfile path and the timezone
    string used throughout this library.
    Returns a time-indexed 1hr-averaged dataframe
    and saves as a csvfile with the station name
    as the name of the csvfile.
    '''
    weather = open(txtfile)
    data = weather.read().split('\n')[1:]
    weather = open(txtfile)
    headings = weather.read().split('\n')[0]
    headings = np.array(headings.split(','))
    weather_df = pd.DataFrame(data[0].split(','))
    weather_df = weather_df.T
    for i in range(1,len(data)):
        weather_df_temp = pd.DataFrame(data[i].split(','))
        weather_df_temp = weather_df_temp.T
        weather_df = pd.concat([weather_df, weather_df_temp])
    weather_df.columns = headings
    station_name = str(weather_df.iloc[4,0])
    weather_df = weather_df.drop(headings[[0,8,12,13,14,15,16,17,18,19,
                    20,21,22,23,24,25,26,27,28]],axis=1)
    weather_df = weather_df.replace('M',np.nan)
    for i in range(1,weather_df.shape[1]):
        weather_df.iloc[:,i] = pd.to_numeric(weather_df.iloc[:,i].values, 
                                             downcast='float')
    weather_df.index = range(0,weather_df.shape[0])
    time = weather_df.valid
    time_list = []
    for i in range(0,len(time)):
        if isFloat(time[i])==False:
            dt_object = datetime.strptime(time[i],'%Y-%m-%d %H:%M')
            dt_object = dt_object.replace(tzinfo = tz.gettz('UTC'))
            dt_object = dt_object.astimezone(tz.gettz(tz_str))
        else:
            dt_object = pd.NaT
        time_list_temp = [pd.Timestamp(dt_object)]
        time_list = time_list + time_list_temp
    non_wind_df = pd.DataFrame(weather_df.iloc[:,[1,2,3,6,7,8]])
    non_wind_df.index = time_list
    non_wind_mean_df = non_wind_df.resample('60T').apply(np.nanmean)
    non_wind_mean_df.columns = ['Temperature','DewPoint','RH',
                                'Precipitation', 'Pressure','Visibility']
    u = -1*(weather_df.iloc[:,5].values)*np.sin(np.deg2rad(weather_df.iloc[:,4].values))
    v = -1*(weather_df.iloc[:,5].values)*np.cos(np.deg2rad(weather_df.iloc[:,4].values))
    ug = -1*(weather_df.iloc[:,9].values)*np.sin(np.deg2rad(weather_df.iloc[:,4].values))
    vg = -1*(weather_df.iloc[:,9].values)*np.cos(np.deg2rad(weather_df.iloc[:,4].values))
    wind_df = pd.DataFrame([u,v,ug,vg])
    wind_df = wind_df.T
    wind_df.columns=['U','V','Ug','Vg']
    wind_df.index = time_list
    wind_mean_df = wind_df.resample('60T').apply(np.nanmean)
    wind_mean_df['WindSpeed'] = np.sqrt(
    (wind_mean_df.U.values**2)+(wind_mean_df.V.values**2))
    wind_mean_df['WindDirection'] = wrapto360(
    np.rad2deg(np.arctan2(-1*wind_mean_df.U.values,
                                                         -1*wind_mean_df.V.values)))
    wind_mean_df['GustSpeed'] = np.sqrt(
    wind_mean_df.Ug.values**2+wind_mean_df.Vg.values**2)
    wind_mean_df['GustDirection'] = wrapto360(
    np.rad2deg(np.arctan2(-1*wind_mean_df.Ug.values,
                                                         -1*wind_mean_df.Vg.values)))
    meteorology_df = pd.DataFrame([non_wind_mean_df.Temperature,
                                  non_wind_mean_df.DewPoint,
                                  non_wind_mean_df.RH,
                                  non_wind_mean_df.Pressure,
                                  non_wind_mean_df.Visibility,
                                  wind_mean_df.WindSpeed,
                                  wind_mean_df.WindDirection,
                                  wind_mean_df.GustSpeed,
                                  wind_mean_df.GustDirection,
                                  non_wind_mean_df.Precipitation]).T
    meteorology_df.Temperature = 5/9*(meteorology_df.Temperature.values-32)
    meteorology_df.DewPoint = 5/9*(meteorology_df.DewPoint.values-32)
    meteorology_df.Pressure = meteorology_df.Pressure.values+1013.25
    meteorology_df.Visibility = meteorology_df.Visibility.values*1609.34
    meteorology_df.WindSpeed = meteorology_df.WindSpeed.values*0.514444
    meteorology_df.GustSpeed = meteorology_df.GustSpeed.values*0.514444
    meteorology_df.Precipitation = meteorology_df.Precipitation.values*25.4
    meteorology_df.to_csv(station_name+'.csv')
    return meteorology_df

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
    ax[0].set_xlabel('Calibration PM$_{2.5}$ ($\mu$g/m$^3$)',
    fontname='Arial', fontsize=20)
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
    ax[1].set_xlabel('Calibration PM$_{2.5}$ ($\mu$g/m$^3$)',
    fontname='Arial', fontsize=20)
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
