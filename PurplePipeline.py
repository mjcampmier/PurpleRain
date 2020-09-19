# built-in
import warnings

# anaconda packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from pandas.plotting import register_matplotlib_converters
from scipy import stats
from scipy import odr

# outside dependencies
import h5py as h5

warnings.filterwarnings("ignore")


def mean_cc(x):
    if len(x) == 0:
        return np.nan
    else:
        n = len(x[~np.isnan(x)]) / len(x)
        if n < 0.8:
            return np.nan
        else:
            return np.nanmean(x)


def axis_lim(y, zero=True):
    mx = np.nanmax(y)
    if mx <= 10:
        ymax = np.ceil(mx / 2) * 2
    elif (mx > 10) and (mx <= 100):
        ymax = np.ceil(mx / 10) * 10
    elif (mx > 100) and (mx <= 300):
        ymax = np.ceil(mx / 20) * 20
    elif (mx > 300) and (mx <= 1000):
        ymax = np.ceil(mx / 50) * 50
    elif (mx > 1000) and (mx <= 2000):
        ymax = np.ceil(mx / 100) * 100
    else:
        ymax = np.ceil(mx / 1000) * 1000
    if zero:
        return [0, ymax]
    else:
        mn = np.abs(np.nanmin(y))
        if np.nanmin(y) < 0:
            sign = -1
        else:
            sign = 1
        if mn <= 10:
            ymin = np.ceil(mn / 2) * 2
        elif (mn > 10) and (mx <= 100):
            ymin = np.ceil(mn / 10) * 10
        elif (mn > 100) and (mn <= 300):
            ymin = np.ceil(mn / 20) * 20
        elif (mn > 300) and (mn <= 1000):
            ymin = np.ceil(mn / 50) * 50
        elif (mn > 1000) and (mn <= 2000):
            ymin = np.ceil(mn / 100) * 100
        else:
            ymin = np.ceil(mn / 1000) * 1000
        return [ymin * sign, ymax]


def purpleair_filter(df, threshold=5, lod=5, upper_cut=100, upper_cut_threshold=0.1, bad_return=np.nan):
    if (upper_cut < lod) or (upper_cut < threshold) or (lod > threshold) or (upper_cut_threshold > 1.0):
        return np.nan
    elif (np.isnan(df.a)) and (np.isnan(df.b)):
        return np.nan
    else:
        if (df.a > upper_cut) or (df.b > upper_cut):
            per_res = (np.abs(df.a - df.b)) / df.a
            if per_res <= upper_cut_threshold:
                return np.nanmean([df.a, df.b])
            else:
                return bad_return
        elif (df.a >= lod) and (df.b >= lod):
            raw_res = np.abs(df.a - df.b)
            if raw_res <= 10:
                return np.nanmean([df.a, df.b])
            else:
                return bad_return
        else:
            return bad_return


def process_bam(csvfile, tzstr):
    BAM = pd.read_csv(csvfile, skiprows=[0, 1, 2, 3], index_col=0)
    BAM = BAM.drop(BAM.columns[3:-1], axis=1)
    BAM.index = pd.to_datetime(BAM.index).tz_localize(tzstr) - pd.Timedelta('1H')
    BAM[BAM['Flow(lpm)'] <= 16.58] = np.nan
    BAM[BAM['ConcHR(ug/m3)'] < 2.4] = np.nan
    BAM[BAM['ConcHR(ug/m3)'] > 2000] = np.nan
    BAM[BAM['ConcRT(ug/m3)'] > 2000] = np.nan
    BAM[BAM['Status'] != 0] = np.nan
    BAM = BAM.drop(['Status', 'ConcRT(ug/m3)', 'Flow(lpm)'], axis=1)
    BAM = BAM.dropna()
    BAM.columns = ['BAM']
    BAM = BAM.resample('60T').apply(np.nanmean)
    return BAM


def file_hdf(sname, time_idx):
    h5file = h5.File(sname, 'w')
    Time = h5file.create_group('Time')
    Time.create_dataset('Time',
                        data=pd.to_datetime(time_idx).to_julian_date())
    return h5file


def fill_hdf(h5file, sensor, date_ind):
    location = h5file.create_group('PurpleAir/' + sensor.name)
    location.create_dataset('Latitude', data=sensor.lat)
    location.create_dataset('Longitude', data=sensor.lon)
    df = pd.DataFrame([sensor.pm25_cf, sensor.relative_humidity,
                       sensor.temperature, sensor.pressure]).T
    df.index = pd.to_datetime(date_ind)
    df.columns = ['pm25', 'rh', 'temp', 'pressure']
    location.create_dataset('PM25', data=df.pm25.values, compression="gzip")
    location.create_dataset('Relative_Humidity', data=df.rh.values, compression="gzip")
    location.create_dataset('Temperature', data=df.temp.values, compression="gzip")
    location.create_dataset('Pressure', data=df.pressure.values, compression="gzip")
    h5file.flush()
    return h5file


def build_hdf(sensor_network, hdf_name, date_ind):
    h5file = file_hdf(hdf_name, date_ind)
    sensor_list = list(sensor_network.network.keys())
    for sensor in sensor_list:
        fill_hdf(h5file, sensor_network.network[sensor], date_ind)
    h5file.close()


class PurpleAir:
    __slots__ = ['name', 'time', 'pm25_cf', 'pm25_cf_B',
                 'temperature', 'relative_humidity', 'pressure', 'lat', 'lon']

    def __init__(self, name, time, pm25_cf_a, pm25_cf_b,
                 temperature, relative_humidity, pressure, lat, lon):
        self.name = name
        try:
            if type(time[:][0]) == np.float64:
                self.time = pd.to_datetime(time, origin='julian', unit='D').round('min').values
            else:
                self.time = time
        except IndexError:
            self.time = np.zeros_like(pm25_cf_a) * np.nan
        self.pm25_cf = pm25_cf_a
        self.pm25_cf_B = pm25_cf_b
        self.temperature = temperature
        self.relative_humidity = relative_humidity
        self.pressure = pressure
        self.lat = lat
        self.lon = lon

    def block_average(self, dt, flags=None):
        df = pd.DataFrame([self.pm25_cf, self.pm25_cf_B,
                           self.temperature, self.relative_humidity, self.pressure])
        df = df.T
        df.columns = ['pm25_cf', 'pm25_cf_B', 'temperature', 'relative_humidity', 'pressure']
        df.index = self.time
        if flags is not None:
            df[flags == 1] = np.nan
        df_block = df.resample(dt).apply(mean_cc)
        df_block = df_block.round(2)
        pa = PurpleAir(self.name, df_block.index.values, df_block.pm25_cf.values,
                       df_block.pm25_cf_B.values, df_block.temperature.values,
                       df_block.relative_humidity.values, df_block.pressure.values,
                       self.lat, self.lon)
        return pa

    def timeseries_plot(self):
        register_matplotlib_converters()
        plt.step(self.time, self.pm25_cf, color=[0.75, 0, 0.75], linewidth=2)
        plt.step(self.time, self.pm25_cf_B, color=[0.35, 0, 0.35], linewidth=2)
        plt.ylim(axis_lim(self.pm25_cf, zero=True))
        plt.xlim([self.time[0], self.time[-1]])
        plt.xticks(rotation=90)
        plt.xlabel('Local Time')
        plt.ylabel('PM$_{2.5}$ ($\mu$g/m$^{3}$)')
        plt.grid()
        plt.legend([self.name + ' A', self.name + ' B'], framealpha=1)

    def bland_altman(self):
        x = np.array([self.pm25_cf, self.pm25_cf_B])
        y = x[0, :] - x[1, :]
        mean_diff = np.nanmean(y)
        std_diff = np.nanstd(y)
        x_lim = axis_lim(x, zero=False)
        y_lim = axis_lim(y, zero=False)
        plt.scatter(np.nanmean(x, 0), y, c='k', s=60)
        plt.hlines(0, x_lim[0], x_lim[1], linestyle='--')
        plt.hlines(mean_diff, x_lim[0], x_lim[1], linewidth=3)
        plt.fill_between(x_lim,
                         mean_diff - 1.96 * std_diff,
                         mean_diff + 1.96 * std_diff,
                         alpha=0.15, color='k')
        plt.grid()
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        plt.xlabel('Interchannel Mean ($\mu$g/m$^{3}$)')
        plt.ylabel('Difference [A-B] ($\mu$g/m$^{3}$)')
        plt.legend(['Interchannel Sample',
                    'Zero Line',
                    'Mean Difference: ' + str(np.round(mean_diff, 2)),
                    'Fence: ' + str([np.round(mean_diff - 1.96 * std_diff, 2),
                                     np.round(mean_diff + 1.96 * std_diff, 2)])],
                   bbox_to_anchor=[1.5, 1], framealpha=1)

    def diurnal_average(self, plot=False):
        df = pd.DataFrame([self.pm25_cf, self.pm25_cf_B])
        df = df.T
        df.columns = ['pm25_cf', 'pm25_cf_B']
        df.mean(axis=1)
        df.index = self.time
        df_med = df.groupby(df.index.hour).apply(np.nanmedian)
        df_iqr = df.groupby(df.index.hour).apply(lambda xp: np.nanpercentile(xp, 75) - np.nanpercentile(xp, 25))
        if plot:
            x = np.linspace(0, 23, 24)
            y1 = df_med - df_iqr.values
            y2 = df_med + df_iqr.values
            plt.plot(x, df_med, color=[0.5, 0, 0.5], linewidth=2)
            plt.fill_between(x, y1, y2, where=y2 >= y1, facecolor=[0.5, 0, 0.5], alpha=0.5)
            plt.xticks(x)
            plt.xlim([0, 23])
            plt.ylim(axis_lim(y2, zero=True))
            plt.xlabel('Hour of Day')
            plt.ylabel('PM$_{2.5}$ ($\mu$g/m$^{3}$)')
            plt.title(self.name + ' Diurnal Trend')
            plt.legend(['Median', 'IQR'], loc='lower left', framealpha=1)
            plt.grid()
        df_diel = pd.concat([df_med, df_iqr], axis=1)
        df_diel.columns = ['Median', 'IQR']
        return df_diel

    def abperformance(self, threshold=5, lod=5, residuals=False):
        mask = ((~np.isnan(self.pm25_cf)) & (~np.isnan(self.pm25_cf_B)))
        pm25 = pd.DataFrame([self.pm25_cf, self.pm25_cf_B]).T
        pm25.index = self.time
        pm25.columns = ['a', 'b']
        filtered = pm25.apply(lambda x: purpleair_filter(x,
                                                         threshold=threshold,
                                                         lod=lod,
                                                         bad_return=-99999), axis=1).values
        down = len(filtered[np.isnan(filtered)]) / len(filtered)
        bad = len(filtered[(filtered == -99999)]) / len(filtered)
        good = 1 - (bad + down)
        if (len(self.pm25_cf[~mask]) < len(self.pm25_cf) - 2) and (
                len(self.pm25_cf_B[~mask]) < len(self.pm25_cf_B) - 2):
            r2 = r2_score(self.pm25_cf[mask], self.pm25_cf_B[mask])
            nrmse = np.sqrt(mean_squared_error(self.pm25_cf[mask],
                                               self.pm25_cf_B[mask])) / np.nanmean(self.pm25_cf_B)
        else:
            r2 = np.nan
            nrmse = np.nan
        if residuals:
            m, b, _, _, _ = stats.linregress(self.pm25_cf[mask], self.pm25_cf_B[mask])
            residuals = (self.pm25_cf * m + b) - self.pm25_cf_B

            fig, ax = plt.subplots(figsize=(20, 10))
            ax1 = plt.subplot(131)
            # n1, _, _ = ax1.hist(error, density=True, color='red')
            ax1.set_xlim([-10, 10])
            # ax1.set_ylim([0, lim10(np.array([n1])) / 20])
            ax1.set_xlabel('Residuals ($\mu$g/m$^{3}$)')
            ax1.set_ylabel('Probability')
            ax1.set_title('Raw Residuals')
            ax1.grid()

            ax2 = plt.subplot(132)
            col = np.arange(0, len(self.time), 1)
            tck = [0,
                   int(np.ceil(np.percentile(col, 25))),
                   int(np.ceil(np.nanmedian(col))),
                   int(np.ceil(np.percentile(col, 75))),
                   int(np.ceil(np.nanmax(col)))]
            cax = ax2.scatter(self.pm25_cf, self.pm25_cf_B, c=col, s=200)
            ax2.plot(self.pm25_cf, self.pm25_cf * m + b, linewidth=2, color='k')
            ax2.set_xlim(axis_lim(np.concatenate([self.pm25_cf, self.pm25_cf_B]), zero=True))
            ax2.set_ylim(axis_lim(np.concatenate([self.pm25_cf, self.pm25_cf_B]), zero=True))
            ax2.set_xlabel('A Channel PM$_{2.5}$ ($\mu$g/m$^{3}$)')
            ax2.set_ylabel('B Channel PM$_{2.5}$ ($\mu$g/m$^{3})$')
            ax2.set_title(self.name)
            ax2.legend(['B = ' + str(np.round(m, 3)) + '*A + ' + str(np.round(b, 3)), '2-min'],
                       loc='upper left', framealpha=1)
            ax2.grid()
            cbar = fig.colorbar(cax, ticks=tck)
            cbar.ax.set_yticklabels(tck)

            ax3 = plt.subplot(133)
            n2, _, _ = ax3.hist(residuals, density=True, color='blue')
            ax3.set_xlim([-10, 10])
            ax3.set_ylim(axis_lim(np.array([n2])) / 20, zero=True)
            ax3.set_xlabel('Residuals ($\mu$g/m$^{3}$)')
            ax3.set_ylabel('Probability')
            ax3.set_title('Model Resiudals')
            plt.grid()

        return down, bad, good, r2, nrmse

    def qf(self, threshold=5, lod=5):
        pm25 = pd.DataFrame([self.pm25_cf, self.pm25_cf_B]).T
        pm25.index = self.time
        pm25.columns = ['a', 'b']
        filtered = pm25.apply(lambda x: purpleair_filter(x, threshold=threshold,
                                                         lod=lod, bad_return=np.nan), axis=1).values
        flag = np.zeros_like(filtered)
        flag[np.isnan(filtered)] = 1
        return flag

    def calibrate(self, source, syy=3):
        models = dict()
        dict_levels = ['PM25', 'PM25-RH', 'PM25-RH-Temp']
        pm25 = np.nanmean(np.array([self.pm25_cf, self.pm25_cf_B]), 0)
        df = pd.DataFrame([pm25, self.relative_humidity, self.temperature])
        df = df.T
        df.index = pd.to_datetime(self.time)
        df.index = df.index.tz_localize(source.index[0].tz)
        df.columns = [self.name, 'RH', 'Temperature']
        df = pd.concat([df, source], axis=1)
        df = df.dropna(how='any')
        dt = int((df.index[1] - df.index[0]).seconds / 3600.0)
        if dt < 1:
            print('Sampling frequency is less than 1 hour, please block average to at least 1 hour and try again.')
        else:
            y = df.iloc[:, -1].values.flatten()
            syy = syy / np.sqrt(dt)
            x = df.iloc[:, [0, 1, 2]].values
            pm25 = x[:, 0]
            sxx_pm25 = np.zeros_like(pm25)
            sxx_pm25[pm25 <= 100] = 10
            sxx_pm25[pm25 > 100] = pm25[[pm25 > 100]] * 0.1
            sxx_rh = np.zeros_like(pm25) + 3
            sxx_tp = np.zeros_like(pm25) + 1
            sxx = np.array([sxx_pm25, sxx_rh, sxx_tp])
            sxx = sxx / np.sqrt(30 * dt)
            sxx = sxx.T
            for i in range(0, 3):
                x_train = x[:, [j for j in range(0, i + 1)]]
                sxx_train = sxx[:, [j for j in range(0, i + 1)]]
                if i == 0:
                    x_train = x_train.flatten()
                    sxx_train = sxx_train.flatten()
                    model = odr.unilinear
                    beta0 = [1, 0]
                else:
                    x_train = x_train.T
                    sxx_train = sxx_train.T
                    model = odr.multilinear
                    beta0 = [0]
                    for j in range(0, i + 1):
                        beta0.insert(0, 1)

                # ODR Fit
                data_odr = odr.Data(x_train, y)
                model_odr = odr.ODR(data_odr, model, beta0=beta0)
                output_odr = model_odr.run()

                # OLS Fit
                data_ols = odr.Data(x_train, y, we=1e-12)
                model_ols = odr.ODR(data_ols, model, beta0=beta0)
                output_ols = model_ols.run()

                # Deming Fit
                data_deming = odr.Data(x_train, y, wd=1. / (sxx_train ** 2), we=1. / (syy ** 2))
                model_deming = odr.ODR(data_deming, model, beta0=beta0)
                output_deming = model_deming.run()

                models[dict_levels[i]] = {'Deming': output_deming,
                                          'ODR': output_odr,
                                          'OLS': output_ols}
            return models


class PurpleAirNetwork:
    __slots__ = ['network']

    def __init__(self):
        self.network = dict()

    def load_h5(self, filename, ftype="archive"):
        f = h5.File(filename, 'r')
        n = dict()
        sensors = list(f['PurpleAir'].keys())
        sensors_directory = ['PurpleAir/' + s + '/' for s in sensors]
        time = f['Time/Time']
        if ftype == 'archive':
            for i in range(0, len(sensors)):
                pm25_cf_a = f[sensors_directory[i] + 'A/PM_CF/PM25_CF'][:]
                pm25_cf_b = f[sensors_directory[i] + 'B/PM_CF/PM25_CF'][:]
                temperature = f[sensors_directory[i] + 'Meteorology/Temperature'][:]
                relative_humidity = f[sensors_directory[i] + 'Meteorology/RH'][:]
                pressure = f[sensors_directory[i] + 'Meteorology/Pressure'][:]
                latitude = f[sensors_directory[i] + 'Latitude']
                longitude = f[sensors_directory[i] + 'Longitude']
                n[sensors[i]] = PurpleAir(sensors[i],
                                          time,
                                          pm25_cf_a,
                                          pm25_cf_b,
                                          temperature,
                                          relative_humidity,
                                          pressure,
                                          latitude,
                                          longitude)
        elif ftype == "qa":
            for i in range(0, len(sensors)):
                latitude = f[sensors_directory[i] + 'Latitude']
                longitude = f[sensors_directory[i] + 'Longitude']
                pm25 = f[sensors_directory[i] + 'PM25'][:]
                relative_humidity = f[sensors_directory[i] + 'Relative_Humidity'][:]
                temperature = f[sensors_directory[i] + 'Temperature'][:]
                pressure = f[sensors_directory[i] + 'Pressure'][:]
                n[sensors[i]] = PurpleAir(sensors[i],
                                          time,
                                          pm25,
                                          np.zeros_like(pm25) * np.nan,
                                          temperature,
                                          relative_humidity,
                                          pressure,
                                          latitude,
                                          longitude
                                          )

        self.network = n
        return self

    def abperformance(self, threshold=5, lod=5, plot=False):
        keys = list(self.network.keys())
        good = np.zeros([len(keys), 1])
        bad = np.zeros([len(keys), 1])
        down = np.zeros([len(keys), 1])
        r2 = np.zeros([len(keys), 1])
        nrmse = np.zeros([len(keys), 1])
        i = 0
        for k in keys:
            d, b, g, r, nr = self.network[k].abperformance(threshold=threshold, lod=lod)
            good[i] = g
            bad[i] = b
            down[i] = d
            r2[i] = r
            nrmse[i] = nr
            i += 1
        ind = np.arange(0, len(keys))
        good = good.flatten()
        bad = bad.flatten()
        down = down.flatten()
        if plot:
            plt.figure(figsize=(10, 5))
            plt.bar(ind, good, color='green')
            plt.bar(ind, bad, bottom=good, color=[207 / 256, 181 / 256, 59 / 256])
            plt.bar(ind, down, bottom=good + bad, color='red')
            plt.xticks(ind, keys, rotation=90)
            plt.xlim([-0.5, len(keys) - 0.5])
            plt.legend(['Good', 'Bad', 'Down'],
                       loc='upper right', bbox_to_anchor=(1.15, 1),
                       framealpha=1)
        return good, bad, down, r2, nrmse

    def block_average(self, dt, flags=None):
        pnet_ba = PurpleAirNetwork()
        n = dict()
        keys = list(self.network.keys())
        for k in keys:
            if flags is not None:
                n[k] = self.network[k].block_average(dt, flags=flags[k].values)
            else:
                n[k] = self.network[k].block_average(dt)
        pnet_ba.network = n
        return pnet_ba

    def diurnal_average(self, plot=False):
        keys = list(self.network.keys())
        df_med = pd.DataFrame(np.zeros([24, len(keys)]) * np.nan, columns=keys)
        df_iqr = pd.DataFrame(np.zeros([24, len(keys)]) * np.nan, columns=keys)
        for k in keys:
            df_s = self.network[k].diurnal_average()
            df_med[k] = df_s.Median
            df_iqr[k] = df_s.IQR
        if plot:
            x = np.linspace(0, 23, 24, dtype=int)
            plt.plot(x, df_med)
            plt.xticks(x, x)
            plt.xlim([0, 23])
            plt.ylim(axis_lim(df_med.values, zero=True))
            plt.xlabel('Hour of Day')
            plt.ylabel('PM$_{2.5}$ ($\mu$g/m$^{3}$)')
            plt.title('Network Diurnal Trends')
            plt.grid()
        return df_med, df_iqr

    def flags(self, threshold=5, lod=5):
        keys = list(self.network.keys())
        time = self.network[keys[0]].time
        flags = pd.DataFrame(np.zeros([len(time), len(keys)]), columns=keys, index=time)
        i = 0
        for k in keys:
            f = self.network[k].qf(threshold=threshold, lod=lod)
            flags.iloc[:, i] = f
            i += 1
        return flags

    def quality_control(self, threshold=5, lod=5, dt='1H', save_file=False, save_name=None):
        flags = self.flags(threshold=threshold, lod=lod)
        pa_net_block_qa = self.block_average(dt, flags=flags)
        keys = list(self.network.keys())
        time_ind = pa_net_block_qa.network[keys[0]].time
        for k in keys:
            sensor = pa_net_block_qa.network[k]
            pm25 = np.nanmean(np.array([sensor.pm25_cf,
                                        sensor.pm25_cf_B]), 0)
            sensor.pm25_cf = pm25
            sensor.pm25_cf_B = np.zeros_like(pm25) * np.nan
        if save_file and save_name:
            build_hdf(pa_net_block_qa, save_name, time_ind)
        return pa_net_block_qa

    def map_network(self):
        keys = list(self.network.keys())
        lat = np.zeros([len(keys), 1])
        lon = np.zeros([len(keys), 1])
        pm25 = np.zeros([len(keys), 1])
        i = 0
        for k in keys:
            lat[i] = self.network[k].lat
            lon[i] = self.network[k].lon
            df_t = pd.DataFrame([self.network[k].pm25_cf, self.network[k].pm25_cf_B]).T.mean(axis=1).mean(axis=0)
            pm25[i] = df_t
            i += 1
        df_map = pd.DataFrame([lon.flatten(), lat.flatten(), pm25.flatten()]).T
        df_map.index = keys
        df_map.columns = ['Longitude', 'Latitude', 'PM2.5']
        return df_map

    def build_df(self, tzstr, drop_b=False):
        keys = list(self.network.keys())
        df = pd.DataFrame(index=self.network[keys[0]].time)
        for k in set(keys):
            df_t = pd.DataFrame([self.network[k].pm25_cf.T,
                                 self.network[k].pm25_cf_B.T,
                                 self.network[k].relative_humidity.T,
                                 self.network[k].temperature.T])
            df_t = df_t.T
            df_t.columns = [k, k + '_B', k + '_RH', k + '_Temperature']
            df_t.index = self.network[k].time
            if drop_b:
                df_t = df_t.drop([df_t.columns[1]], axis=1)
            df = pd.concat([df, df_t], axis=1)
        df.index = df.index.tz_localize(tzstr)
        return df
