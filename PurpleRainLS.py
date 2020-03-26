"""
Author: Mark Campmier
Github/Twitter: @mjcampmier
Last Edit: 22 March 2020
"""
# built-in
import os
import glob
import warnings
from json import JSONDecodeError
import requests
from pathlib import Path

# anaconda packages
import numpy as np
import pandas as pd
from scipy import io

# outside dependencies
import h5py as h5

warnings.filterwarnings("ignore")


def build_dir(dir_path):
    dir_path = os.path.join(os.getcwd(), dir_path)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


def download_database():
    success = False
    count = 0
    while count < 10:
        try:
            r = requests.get('https://www.purpleair.com/json')
            all_pa = dict(r.json())
            df_pa = pd.DataFrame(all_pa['results'])
            df_pa.drop(['AGE', 'A_H', 'DEVICE_LOCATIONTYPE',
                        'Flag', 'Hidden',
                        'ID', 'LastSeen', 'Ozone1',
                        'PM2_5Value', 'ParentID',
                        'Stats', 'Type', 'Voc',
                        'humidity', 'isOwner',
                        'pressure', 'temp_f',
                        ], axis=1, inplace=True)
        except:
            count += 1
            success = False
        else:
            success = True
            print('Database successfully scraped.')
            count = 100
    if success is True:
        return df_pa
    else:
        print('Error connecting to PurpleAir server, check internet connection and retry in 2 minutes.')


def sensor_metadata(df_pa, sensor):
    sensor_A = sensor
    sensor_B = sensor + ' B'
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
        LAT = float(df_pa.Lat[df_pa.Label == sensor_A].item())
        LON = float(df_pa.Lon[df_pa.Label == sensor_A].item())
    except ValueError:
        print('Name not found. Please check your PurpleAir registration for: ', sensor)
        ID = [np.nan, np.nan, np.nan, np.nan]
        KEYS = [np.nan, np.nan, np.nan, np.nan]
        LAT = np.nan
        LON = np.nan
    return ID, KEYS, LAT, LON


def download_request(id_, key, sd, ed, fname, down_dir):
    SD_list = sd.split('-')
    ED_list = ed.split('-')
    sd = pd.Timestamp(int(SD_list[0]),
                      int(SD_list[1]),
                      int(SD_list[2]))
    ed = pd.Timestamp(int(ED_list[0]),
                      int(ED_list[1]),
                      int(ED_list[2]) + 1)
    date_ind = pd.date_range(sd, ed, freq='1D')
    dir_path = build_dir(os.path.join(down_dir, fname))
    for i in range(0, len(date_ind)):
        if i != len(date_ind) - 1:
            SDi = str(date_ind[i]).split(' ')[0]
            EDi = str(date_ind[i + 1]).split(' ')[0]
            url = 'https://thingspeak.com/channels/' + id_ + '/feed.csv?api_key=' + key + '&offset=0&average=0&round' \
                                                                                          '=2&start=' + SDi + \
                  '%2000:00:SS&end=' + EDi + '%2000:00:00 '
            r = requests.get(url, allow_redirects=True)
            open(SDi + '.csv', 'wb').write(r.content)
            os.rename(SDi + '.csv', dir_path + '/' + SDi + '.csv')
    flist = glob.glob(dir_path + '/*.csv')
    full_df = pd.read_csv(flist[0])
    for i in range(1, len(flist)):
        temp_df = pd.read_csv(flist[i])
        full_df = pd.concat([full_df, temp_df])
    full_df = full_df.sort_values(by=['created_at'])
    full_name = down_dir + '/' + fname + '.csv'
    full_df.to_csv(full_name, index=False)
    return full_name


def assign_field_names(fname, fields):
    df_csv = pd.read_csv(fname)
    if len(df_csv.columns) != len(fields):
        del fields[1]
        df_csv.columns = fields
    else:
        df_csv.columns = fields
    df_csv.to_csv(fname, index=False)


def download_sensor(sensor, sd, ed, down_dir, db=None):
    if db is None:
        db = []
    if len(db) == 0:
        df_pa = download_database()
    else:
        df_pa = db
    ID, KEYS, LAT, LON = sensor_metadata(df_pa, sensor)
    sensor_A = sensor.replace(' ', '_')
    sensor_B = sensor_A + '_B'
    sd = sd.replace('-', '_')
    ed = ed.replace('-', '_')
    fname = ['Primary_' + sensor_A + '_' + sd + '_' + ed,
             'Secondary_' + sensor_A + '_' + sd + '_' + ed,
             'Primary_' + sensor_B + '_' + sd + '_' + ed,
             'Secondary_' + sensor_B + '_' + sd + '_' + ed]
    fields_primary_A = ["created_at", "entry_id", "PM1.0_CF1_ug/m3",
                        "PM2.5_CF1_ug/m3", "PM10.0_CF1_ug/m3", "UptimeMinutes",
                        "ADC", "Temperature_F", "Humidity_%", "PM2.5_ATM_ug/m3"]
    fields_secondary_A = ["created_at", "entry_id",
                          ">=0.3um/dl", ">=0.5um/dl", ">1.0um/dl", ">=2.5um/dl",
                          ">=5.0um/dl", ">=10.0um/dl", "PM1.0_ATM_ug/m3", "PM10_ATM_ug/m3"]
    fields_primary_B = ["created_at", "entry_id", "PM1.0_CF1_ug/m3",
                        "PM2.5_CF1_ug/m3", "PM10.0_CF1_ug/m3", "UptimeMinutes",
                        "RSSI_dbm", "Pressure_hpa", "IAQ", "PM2.5_ATM_ug/m3"]
    fields_secondary_B = ["created_at", "entry_id",
                          ">=0.3um/dl", ">=0.5um/dl", ">1.0um/dl", ">=2.5um/dl",
                          ">=5.0um/dl", ">=10.0um/dl", "PM1.0_ATM_ug/m3", "PM10_ATM_ug/m3"]
    fields = [fields_primary_A, fields_secondary_A,
              fields_primary_B, fields_secondary_B]
    for i in range(0, len(ID)):
        full_name = download_request(ID[i], KEYS[i], sd, ed, fname[i], down_dir)
        assign_field_names(full_name, fields[i])
    print('Successfully downloaded ' + str(sensor))
    return LAT, LON


def time_master(dfpa, dfsa, dfpb, dfsb, tzstr, date_ind):
    dfpa_ind = pd.to_datetime(dfpa.Time, utc=True)
    dfsa_ind = pd.to_datetime(dfsa.Time, utc=True)
    dfpb_ind = pd.to_datetime(dfpb.Time, utc=True)
    dfsb_ind = pd.to_datetime(dfsb.Time, utc=True)
    dfpa = dfpa.drop(['Time'], axis=1)
    dfsa = dfsa.drop(['Time'], axis=1)
    dfpb = dfpb.drop(['Time'], axis=1)
    dfsb = dfsb.drop(['Time'], axis=1)
    dfpa.index = dfpa_ind
    dfsa.index = dfsa_ind
    dfpb.index = dfpb_ind
    dfsb.index = dfsb_ind
    dfpa.index = dfpa.index.tz_convert(tzstr)
    dfsa.index = dfsa.index.tz_convert(tzstr)
    dfpb.index = dfpb.index.tz_convert(tzstr)
    dfsb.index = dfsb.index.tz_convert(tzstr)
    df_summary = dfpa.merge(dfsa, how='outer', right_index=True, left_index=True)
    df_summary = df_summary.merge(dfpb, how='outer', right_index=True, left_index=True)
    df_summary = df_summary.merge(dfsb, how='outer', right_index=True, left_index=True)
    df_summary = df_summary.resample('2T').apply(np.nanmean)
    df_summary = df_summary.reindex(date_ind)
    return df_summary


def file_hdf(sname, time_idx):
    h5file = h5.File(sname + '.h5', 'w')
    Time = h5file.create_group('Time')
    Time.create_dataset('Time', data=time_idx)
    return h5file


def fill_hdf(h5file, sensor, df, lat, lon):
    sensor = sensor.split('\\')[-1]
    location = h5file.create_group('PurpleAir/' + sensor)
    a_raw = h5file.create_group('PurpleAir/' + sensor + '/A/PM_Raw')
    a_cf = h5file.create_group('PurpleAir/' + sensor + '/A/PM_CF')
    a_counts = h5file.create_group('PurpleAir/' + sensor + '/A/Counts')
    b_raw = h5file.create_group('PurpleAir/' + sensor + '/B/PM_Raw')
    b_cf = h5file.create_group('PurpleAir/' + sensor + '/B/PM_CF')
    b_counts = h5file.create_group('PurpleAir/' + sensor + '/B/Counts')
    met = h5file.create_group('PurpleAir/' + sensor + '/Meteorology')
    try:
        a_raw.create_dataset('PM1_Raw', data=df['PM1_Raw_A'].values)
    except:
        print('Failed to fill: ' + sensor)
    else:
        location.create_dataset('Latitude', data=lat)
        location.create_dataset('Longitude', data=lon)
        a_raw.create_dataset('PM25_Raw', data=df['PM25_Raw_A'].values, compression="gzip")
        a_raw.create_dataset('PM10_Raw', data=df['PM10_Raw_A'].values, compression="gzip")
        a_cf.create_dataset('PM1_CF', data=df['PM1_CF_A'].values, compression="gzip")
        a_cf.create_dataset('PM25_CF', data=df['PM25_CF_A'].values, compression="gzip")
        a_cf.create_dataset('PM10_CF', data=df['PM10_CF_A'].values, compression="gzip")
        b_raw.create_dataset('PM1_Raw', data=df['PM1_Raw_B'].values, compression="gzip")
        b_raw.create_dataset('PM25_Raw', data=df['PM25_Raw_B'].values, compression="gzip")
        b_raw.create_dataset('PM10_Raw', data=df['PM10_Raw_B'].values, compression="gzip")
        b_cf.create_dataset('PM1_CF', data=df['PM1_CF_B'].values, compression="gzip")
        b_cf.create_dataset('PM25_CF', data=df['PM25_CF_B'].values, compression="gzip")
        b_cf.create_dataset('PM10_CF', data=df['PM10_CF_B'].values, compression="gzip")
        met.create_dataset('Temperature', data=df['Temperature_A'].values, compression="gzip")
        met.create_dataset('Pressure', data=df['Pressure_B'].values, compression="gzip")
        met.create_dataset('RH', data=df['RH_A'].values, compression="gzip")
        a_counts.create_dataset('PM03_dl', data=df['PM03_dl_A'].values, compression="gzip")
        a_counts.create_dataset('PM05_dl', data=df['PM05_dl_A'].values, compression="gzip")
        a_counts.create_dataset('PM1_dl', data=df['PM1_dl_A'].values, compression="gzip")
        a_counts.create_dataset('PM25_dl', data=df['PM25_dl_A'].values, compression="gzip")
        a_counts.create_dataset('PM10_dl', data=df['PM10_dl_A'].values, compression="gzip")
        b_counts.create_dataset('PM03_dl', data=df['PM03_dl_B'].values, compression="gzip")
        b_counts.create_dataset('PM05_dl', data=df['PM05_dl_B'].values, compression="gzip")
        b_counts.create_dataset('PM1_dl', data=df['PM1_dl_B'].values, compression="gzip")
        b_counts.create_dataset('PM25_dl', data=df['PM25_dl_B'].values, compression="gzip")
        b_counts.create_dataset('PM10_dl', data=df['PM10_dl_B'].values, compression="gzip")
    h5file.flush()
    return h5file


def build_hdf(name_list, hdfname, tzstr, date_ind, lat, lon):
    h5file = file_hdf(hdfname, np.array(date_ind.to_julian_date()))
    sensors = []
    for i in range(0, len(name_list)):
        try:
            pa = name_list[i][0]
        except:
            print('no data from IDX: ' + str(i))
        else:
            try:
                pb = name_list[i][1]
                sa = name_list[i][2]
                sb = name_list[i][3]
            except:
                pb = pa
                sb = sa
                no_b = True
            else:
                no_b = False
            sensors.append(name_list[i][0].replace('Primary_', '').split('_20')[0])
            pa = pd.read_csv(Path(pa), skip_blank_lines=False)
            sa = pd.read_csv(Path(sa), skip_blank_lines=False)
            pb = pd.read_csv(Path(pb), skip_blank_lines=False)
            sb = pd.read_csv(Path(sb), skip_blank_lines=False)
            lpa = ['Time', 'entry_id', 'PM1_Raw_A', 'PM25_Raw_A', 'PM10_Raw_A',
                   'Uptime', 'ADC', 'Temperature_A', 'RH_A', 'PM25_CF_A']
            lpb = ['Time', 'entry_id', 'PM1_Raw_B', 'PM25_Raw_B', 'PM10_Raw_B',
                   'Uptime', 'ADC', 'Pressure_B', '__', 'PM25_CF_B']
            lsa = ['Time', 'entry_id', 'PM03_dl_A', 'PM05_dl_A', 'PM1_dl_A',
                   'PM25_dl_A', 'PM5_dl_A', 'PM10_dl_A', 'PM1_CF_A', 'PM10_CF_A']
            lsb = ['Time', 'entry_id', 'PM03_dl_B', 'PM05_dl_B', 'PM1_dl_B',
                   'PM25_dl_B', 'PM5_dl_B', 'PM10_dl_B', 'PM1_CF_B', 'PM10_CF_B']
            dpa = ['entry_id', 'Uptime', 'ADC']
            dpb = ['entry_id', 'Uptime', 'ADC', '__']
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
                pa.iloc[0, 0] = '1996-10-17 19:00:00 UTC'
                pb.iloc[0, 0] = '1996-10-17 19:00:00 UTC'
                sa.iloc[0, 0] = '1996-10-17 19:00:00 UTC'
                sb.iloc[0, 0] = '1996-10-17 19:00:00 UTC'
            if no_b:
                pb.iloc[:, 1:] = np.nan
                sb.iloc[:, 1:] = np.nan
            df_summary = time_master(pa, sa, pb, sb, tzstr, date_ind)
            h5file = fill_hdf(h5file, str(sensors[i].split("'/'")[0].replace('_', ' ')), df_summary, lat[i], lon[i])
            print("Filled HDF for " + str(sensors[i].split("'/'")[0].replace('_', ' ')))
    h5file.close()


def load_dict_from_hdf5(filename):
    with h5.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def hdf5_to_mat(hdfile):
    data = load_dict_from_hdf5(hdfile)
    arr = np.array(list(data.items()))
    io.savemat(hdfile[:-3] + '.mat', {'arr': arr})
    print('Successfully built .mat file.')


def sensors_from_csv(csvfile):
    sensors = pd.read_csv(csvfile)
    sensor_list = sensors.iloc[:, 0].values.tolist()
    return sensor_list


def downloaded_file_list(directory, sensor_list):
    name_list = []
    for i in range(0, len(sensor_list)):
        search_name = os.path.join(directory, '*_' + sensor_list[i].replace(' ', '_') + '_*.csv')
        name_list_temp = glob.glob(search_name)
        for j in range(0, len(name_list_temp)):
            name_list_temp[j] = name_list_temp[j].split('/')[-1]
        name_list.append(sorted(name_list_temp))
    names = name_list
    return names


def h5file_query(h5file, query_string):
    f = h5.File(h5file, 'r')
    array = f[query_string][:]
    return array


def download_list(sensor_list_file, sd, ed, hdfname, tz):
    dir_name = build_dir(hdfname)
    sensor_list = pd.read_csv(sensor_list_file, header=None)
    sensor_list = sensor_list.iloc[:, 0]
    df_db = download_database()
    LAT, LON = [], []
    for i in range(0, len(sensor_list)):
        lat, lon = download_sensor(sensor_list[i], sd, ed, hdfname, db=df_db)
        LAT.append(lat)
        LON.append(lon)
    names = downloaded_file_list(dir_name, sensor_list.tolist())
    sd = sd.split('-')
    ed = ed.split('-')
    start_date = pd.Timestamp(year=int(sd[0]),
                              month=int(sd[1]),
                              day=int(sd[2]),
                              hour=0, minute=0, second=0,
                              tz='UTC')
    end_date = pd.Timestamp(year=int(ed[0]),
                            month=int(ed[1]),
                            day=int(ed[2]),
                            hour=23, minute=59, second=59,
                            tz='UTC')
    date_ind = pd.date_range(start_date, end_date, freq='2T')
    date_ind = date_ind.tz_convert(tz)
    build_hdf(names, hdfname, tz, date_ind, LAT, LON)
    hdf5_to_mat(hdfname + '.h5')
    print('Successfully downloaded all sensors.')
