'''
Author: Mark Campmier
Github/Twitter: @mjcampmier
Last Edit: 22 March 2020
'''

# built-in
import warnings

# anaconda packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from pandas.plotting import register_matplotlib_converters
from scipy import stats

# outside dependencies
import h5py as h5

warnings.filterwarnings("ignore")


def lim10(x):
    return np.nanmax(np.ceil(x / 10) * 10)


class PurpleAir:
    def __init__(self, name, time, pm25_cf_a, pm25_cf_b,
                 temperature, relative_humidity, pressure, lat, lon):
        self.name = name
        if type(time[:][0]) == np.float64:
            self.time = pd.to_datetime(time, origin='julian', unit='D').values
        else:
            self.time = time
        self.pm25_cf_A = pm25_cf_a
        self.pm25_cf_B = pm25_cf_b
        self.temperature = temperature
        self.relative_humidity = relative_humidity
        self.pressure = pressure
        self.lat = lat
        self.lon = lon

    def block_average(self, dt):
        df = pd.DataFrame([self.pm25_cf_A, self.pm25_cf_B,
                           self.temperature, self.relative_humidity, self.pressure])
        df = df.T
        df.columns = ['pm25_cf_A', 'pm25_cf_B', 'temperature', 'relative_humidity', 'pressure']
        df.index = self.time
        df_block = df.resample(dt).apply(np.nanmean)
        pa = PurpleAir(self.name, df_block.index.values, df_block.pm25_cf_A.values,
                       df_block.pm25_cf_B.values, df_block.temperature.values,
                       df_block.relative_humidity.values, df_block.pressure.values,
                       self.lat, self.lon)
        return pa

    def timeseries_plot(self):
        register_matplotlib_converters()
        plt.step(self.time, self.pm25_cf_A, color=[0.75, 0, 0.75], linewidth=2)
        plt.step(self.time, self.pm25_cf_B, color=[0.35, 0, 0.35], linewidth=2)
        plt.ylim([0, lim10(self.pm25_cf_A)])
        plt.xlim([self.time[0], self.time[-1]])
        plt.xticks(rotation=90)
        plt.xlabel('Local Time')
        plt.ylabel('PM$_{2.5}$ ($\mu$g/m$^{3}$)')
        plt.grid()
        plt.legend([self.name + ' A', self.name + ' B'], framealpha=1)

    def diurnal_average(self, plot=False):
        df = pd.DataFrame([self.pm25_cf_A, self.pm25_cf_B])
        df = df.T
        df.columns = ['pm25_cf_A', 'pm25_cf_B']
        df.mean(axis=1)
        df.index = self.time
        df_med = df.groupby(df.index.hour).apply(np.nanmedian)
        df_iqr = df.groupby(df.index.hour).apply(lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25))
        if plot == True:
            x = np.linspace(0, 23, 24)
            y = df_med.values
            y1 = df_med - df_iqr.values
            y2 = df_med + df_iqr.values
            plt.plot(x, df_med, color=[0.5, 0, 0.5], linewidth=2)
            plt.fill_between(x, y1, y2, where=y2 >= y1, facecolor=[0.5, 0, 0.5], alpha=0.5)
            plt.xticks(x)
            plt.xlim([0, 23])
            plt.ylim([0, lim10(y2)])
            plt.xlabel('Hour of Day')
            plt.ylabel('PM$_{2.5}$ ($\mu$g/m$^{3}$)')
            plt.title(self.name + ' Diurnal Trend')
            plt.legend(['Median', 'IQR'], loc='lower left', framealpha=1)
            plt.grid()
        df_diel = pd.concat([df_med, df_iqr], axis=1)
        df_diel.columns = ['Median', 'IQR']
        return df_diel

    def abperformance(self, residuals=False):
        error = self.pm25_cf_A - self.pm25_cf_B
        mask = ((~np.isnan(self.pm25_cf_A)) & (~np.isnan(self.pm25_cf_B)))
        pm25 = pd.DataFrame([self.pm25_cf_A, self.pm25_cf_B]).mean(axis=0).T.values
        down = len(pm25[np.isnan(pm25)]) / len(pm25)
        bad = len(pm25[(error < -5) | (error > 5)]) / len(pm25)
        good = 1 - (bad + down)
        if (len(self.pm25_cf_A[~mask]) < len(self.pm25_cf_A) - 2) and (
                len(self.pm25_cf_B[~mask]) < len(self.pm25_cf_B) - 2):
            r2 = r2_score(self.pm25_cf_A[mask], self.pm25_cf_B[mask])
            nrmse = np.sqrt(mean_squared_error(self.pm25_cf_A[mask],
                                               self.pm25_cf_B[mask])) / np.nanmean(self.pm25_cf_B)
        else:
            r2 = np.nan
            nrmse = np.nan
        if residuals == True:
            m, b, _, _, _ = stats.linregress(self.pm25_cf_A[mask], self.pm25_cf_B[mask])
            residuals = (self.pm25_cf_A * m + b) - self.pm25_cf_B

            fig, ax = plt.subplots(figsize=(20, 10))
            ax1 = plt.subplot(131)
            n1, _, _ = ax1.hist(error, density=True, color='red')
            ax1.set_xlim([-10, 10])
            ax1.set_ylim([0, lim10(np.array([n1])) / 20])
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
            cax = ax2.scatter(self.pm25_cf_A, self.pm25_cf_B, c=col, s=200)
            ax2.plot(self.pm25_cf_A, self.pm25_cf_A * m + b, linewidth=2, color='k')
            ax2.set_xlim([0, lim10(np.concatenate([self.pm25_cf_A, self.pm25_cf_B]))])
            ax2.set_ylim([0, lim10(np.concatenate([self.pm25_cf_A, self.pm25_cf_B]))])
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
            ax3.set_ylim([0, lim10(np.array([n2])) / 20])
            ax3.set_xlabel('Residuals ($\mu$g/m$^{3}$)')
            ax3.set_ylabel('Probability')
            ax3.set_title('Model Resiudals')
            plt.grid()

        return down, bad, good, r2, nrmse

    def qf(self):
        error = np.abs(self.pm25_cf_B - self.pm25_cf_A)
        flag = np.zeros((len(error), 1))
        flag[(error > 5)] = 1
        return flag


class PurpleAirNetwork:
    def __init__(self):
        self.network = dict()

    def load_h5(self, filename):
        f = h5.File(filename, 'r')
        n = dict()
        sensors = list(f['PurpleAir'].keys())
        sensors_directory = ['PurpleAir/' + s + '/' for s in sensors]
        time = f['Time/Time']
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
        self.network = n
        return self

    def abperformance(self, plot=False):
        keys = list(self.network.keys())
        good = np.zeros([len(keys), 1])
        bad = np.zeros([len(keys), 1])
        down = np.zeros([len(keys), 1])
        r2 = np.zeros([len(keys), 1])
        nrmse = np.zeros([len(keys), 1])
        i = 0
        for k in keys:
            d, b, g, r, nr = self.network[k].abperformance()
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
        if plot == True:
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

    def block_average(self, dt):
        pnet_ba = PurpleAirNetwork()
        n = dict()
        keys = list(self.network.keys())
        for k in keys:
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
        if plot == True:
            x = np.linspace(0, 23, 24, dtype=int)
            plt.plot(x, df_med)
            plt.xticks(x, x)
            plt.xlim([0, 23])
            plt.ylim([0, lim10(df_med.values)])
            plt.xlabel('Hour of Day')
            plt.ylabel('PM$_{2.5}$ ($\mu$g/m$^{3}$)')
            plt.title('Network Diurnal Trends')
            plt.grid()
        return df_med, df_iqr

    def flags(self):
        keys = list(self.network.keys())
        time = self.network[keys[0]].time
        flags = pd.DataFrame(np.zeros([len(time), len(keys)]), columns=keys, index=time)
        i = 0
        for k in keys:
            f = self.network[k].qf()
            flags.iloc[:, i] = f
            i += 1
        return flags

    def map_network(self):
        keys = list(self.network.keys())
        lat = np.zeros([len(keys), 1])
        lon = np.zeros([len(keys), 1])
        pm25 = np.zeros([len(keys), 1])
        i = 0
        for k in keys:
            lat[i] = self.network[k].lat
            lon[i] = self.network[k].lon
            df_t = pd.DataFrame([self.network[k].pm25_cf_A, self.network[k].pm25_cf_B]).T.mean(axis=1).mean(axis=0)
            pm25[i] = df_t
            i += 1
        df_map = pd.DataFrame([lon.flatten(), lat.flatten(), pm25.flatten()]).T
        df_map.index = keys
        df_map.columns = ['Longitude', 'Latitude', 'PM2.5']
        return df_map