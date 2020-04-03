#!/usr/bin/env python
"""Some functionality for playing around with the meteo data"""
import calendar
import datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import pandas as pd

from scipy.interpolate import UnivariateSpline

import matplotlib.dates as mdates

import ipywidgets as widgets

from IPython.display import display

from process_meteo_drivers import grab_meteo_data

parameters = ["Irradiance \n(kJ·m-2·d-1)",
     "Min temp \n(degC)",
    "Max temp \n (degC)",
    "Vap press\n(kPa)",
    "Wind spd \n (m·s-1)",
    "Precip\n (mm·d-1)"]


regions = ["Ashanti",  "Brong_Ahafo",  "Central",  
           "Eastern",  "Greater_Accra",  "Northern",
           "Upper_East",  "Upper_West",  "Volta",
           "Western"]


def water_limitation(precip, et):
    return np.where(precip >= et, 1, precip/et)


def temp_constraint(temp, t_min=12, t_max=41, t_opt=28):
    f_temp = np.zeros_like(temp)
    f_temp[temp < t_min] = 0.
    f_temp = np.where(np.logical_and(t_min <= temp, temp <= t_opt),
             (temp - t_min)/(t_opt - t_min),
             f_temp)
    f_temp = np.where(temp >= t_opt,
             (t_max - temp)/(t_max-t_opt),
             f_temp)
    f_temp[f_temp<0] = 0.
    return f_temp

def aggregate_plots():
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2010, 12, 31)

    dates = pd.date_range(start_date, end_date, freq='D')

    options = [(date.strftime(' %d %b %Y '), date) for date in dates]
    index = (0, len(options)-1)

    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Sowing & Harvest',
        orientation='horizontal',
        layout={'width': '600px'}
    )

    def plot_aggr_meteo(sowing_harvesting, region_name, selected_years):
        sowing, harvesting = sowing_harvesting
        meteo_files = get_region_data_func(region_name, selected_years, do_plot=False)
        data = aggregate_meteo(meteo_files, sowing, harvesting, aggr=np.sum)
        fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True,
                figsize=(12,12), squeeze=True)
        axs = axs.flatten()
        for i in range(6):
            axs[i].plot(data[:, 0], data[:, i + 1], '-o')
            axs[i].set_title(parameters[i])
        fig.suptitle(region_name)

    widgets.interact_manual(
        plot_aggr_meteo,
        sowing_harvesting=selection_range_slider,
        region_name=widgets.Dropdown(
                        options=regions, value='Central', description='Region:',
                        disabled=False,), 
        selected_years=widgets.IntRangeSlider(min=2010, max=2018, value=(2015,2016)))



def plot_meteo(meteo):
    """Plot WOFOST meteo files
    
    Parameters
    ----------
    meteo : str or iter
        Set of text files that contain different variables of interest
        to plot. Can be done with just a file or a list of files.
    """

    if type(meteo) != type([]): meteo = [meteo]
    
    fig, axs = plt.subplots(nrows=3, ncols=2, 
                figsize=(13,9), sharex=True,squeeze=True)
    axs = axs.flatten()
    for meteo_file in meteo:
        d = np.loadtxt(meteo_file.as_posix(), skiprows=20)
        for i,p in enumerate(parameters):
            if i == 5:
                axs[i].plot(d[:,2], d[:,3+i], '-', lw=0.8, label=meteo_file.name)
            else:
                axs[i].plot(d[:,2], d[:,3+i], '-', lw=0.8)
            axs[i].set_ylabel(p, fontsize=9)
    axs[-1].legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()


def calc_et0(r_surf, t_min, t_max):
    """Calculate Hargreaves ET0 in mm/day
    
    Parameters
    ----------
    r_surf : float, array
        Surface radiation
    t_min : float, array
        Min daily temperature (degC)
    t_max : float array
        Max daily temperature (degC)
    """
    t_mean = 0.5*(t_min + t_max)
    lam = 2260.
    et0 = 0.0023 * ((t_max - t_min)** 0.5) * (t_mean + 17.8) * r_surf / lam
    return et0



def aggregate_meteo(meteo_files, sowing, harvesting, aggr=np.cumsum):
    rr = []
    for meteo_file in meteo_files:
        year = int(meteo_file.name.split(".")[-1])
        d = np.loadtxt(meteo_file.as_posix(), skiprows=20)
        doy = np.array([
            dt.datetime(year, 1, 1) + dt.timedelta(days=int(j))
            for j in d[:, 2]])
        sow = dt.datetime(year, sowing.month, sowing.day)
        harvest = dt.datetime(year, harvesting.month, harvesting.day)

        passer = np.logical_and(doy >= sow,
                                doy <= harvest)
        xx = aggr(d[passer, 3:], axis=0)
        rr.append(np.r_[year, xx])
    rr = np.array(rr)
    return rr


def extract_data(lat, lon, meteo_folder="era5_data",
                 n_threads=2):
    meteo_files = []
    wrapper = lambda year: grab_meteo_data(
            lon,
            lat,
            year,
            f"era5_africa_{year:d}.nc",
            meteo_folder,
            site_name="Ghana")
    # create a thread pool of n_threads threads

    years = np.arange(2010,2019).astype(np.int)
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for _ in executor.map(wrapper, years):
            pass


def get_region_data():
    @widgets.interact(region_name=widgets.Dropdown(
                            options=regions, value='Central', description='Region:',
                            disabled=False,), 
                            selected_years=widgets.IntRangeSlider(min=2010, max=2018, value=(2015,2016)))
    def get_region_data_fun(region_name, selected_years, do_plot=True):
        start_year, end_year = selected_years
        meteo_files = sorted([f for f in Path(f'./data/meteo/{region_name}/').glob(f"{region_name}.20??")])
        years = [int(f.name.split(".")[1]) for f in meteo_files]
        do_files = [f for y, f in zip(years, meteo_files) if start_year <= y <= end_year]
        if do_plot:
            plot_meteo(do_files)
        else:
            return do_files
        
        
def get_region_data_func(region_name, selected_years, do_plot=True):
    start_year, end_year = selected_years
    meteo_files = sorted([f for f in Path(f'./data/meteo/{region_name}/').glob(f"{region_name}.20??")])
    years = [int(f.name.split(".")[1]) for f in meteo_files]
    do_files = [f for y, f in zip(years, meteo_files) if start_year <= y <= end_year]
    if do_plot:
        plot_meteo(do_files)
    else:
        return do_files

def plot_stressors_func(region_name, year):
    
    meteo_file = sorted([f for f in Path(f'./data/meteo/{region_name}/').glob(f"{region_name}.{year}")])[0]
    df = pd.read_csv(meteo_file, skiprows=20, sep="\t", 
                 names=["station", "year", "doy", "irradiance", "tmin", "tmax", "vpd", "mws", "prec"])
    df.set_index(df['doy'])
    et0 = calc_et0(df.irradiance.values, df.tmin.values, df.tmax.values)
    f_water = water_limitation(df.prec.values, et0)
    f_temp = temp_constraint(df.tmax.values)
    plt.figure(figsize=(15, 5))
    plt.plot(df.doy, f_water, 'o', lw=0.5, mfc="none", label="Water stress")
    plt.plot(df.doy, f_temp, 's', lw=0.5, mfc="none", label="Heat stress")
    plt.plot(df.doy, np.convolve(f_water, np.ones(10)/10., mode="same"), '-', lw=3, label="Smoothed Water Stress")
    plt.plot(df.doy, np.convolve(f_temp, np.ones(10)/10., mode="same"), '-', lw=3, label="Smoothed Heat Stress")
    plt.legend(loc="best")

def plot_stressors():
    widgets.interact(plot_stressors_func,region_name=widgets.Dropdown(
                        options=regions, value='Upper_East', description='Region:',
                        disabled=False,), 
                year=widgets.IntSlider(min=2010, max=2018, value=(2015)))



def meteo_calculations(year):
    df = pd.read_csv(f"data/meteo/-022611_106965/-022611_106965.{year}", skiprows=20, sep="\t", 
                     names=["station", "year", "doy", "irradiance", "tmin", "tmax", "vpd", "mws", "prec"])
    df.set_index(df['doy'])
    et0 = calc_et0(df.irradiance.values, df.tmin.values, df.tmax.values)
    f_water = water_limitation(df.prec.values, et0)
    f_temp = temp_constraint(df.tmax.values)
    f_water = np.convolve(f_water, np.ones(10)/10, mode="same")
    f_temp = np.convolve(f_temp, np.ones(5)/5, mode="same")
    return f_water, f_temp


def extract_smooth_fapar(product="fapar", year=2018, smoother=100):
    golden_ratio = 0.61803398875
    mask57 = 0b11100000  # Select bits 5, 6 and 7
    product = product.lower()
    if calendar.isleap(year):
        xs = np.arange(1, 367)
    else:
        xs = np.arange(1, 366)
    
    year = year-2003
    x = np.arange(1, 366, 8)
    y = np.loadtxt(f"data/mcd15_{product}_2003_2018_-022611_106965.txt")[:, year]
    qa = np.loadtxt("data/mcd15_qa_2003_2018_-022611_106965.txt", dtype=np.uint8)[:, year]
    unc = np.power(golden_ratio, np.right_shift(np.bitwise_and(qa, mask57), 5).astype(np.float32))
    spl = UnivariateSpline(x, y, w=(1./unc)**2)
    spl.set_smoothing_factor(smoother)
    return spl(xs)
    #plt.plot(x, y)
    #plt.plot(xs, spl(xs), 'b', lw=3)


def crop_model():
    
    start_date = dt.datetime(2014, 2, 1)
    end_date = dt.datetime(2014, 12, 1)

    dates = pd.date_range(start_date, end_date, freq='D')

    options = [(date.strftime(' %d %b '), date) for date in dates]
    index = (0, len(options) - 1)
    integration_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Sowing & Harvest',
        orientation='horizontal',
        layout={'width': '600px'}
    )
    year_widget = widgets.IntSlider(min=2010, max=2018, value=2015)
    widgets.interact(crop_model_func, year=year_widget,
                 integration_time = integration_slider,
                 epsilon = widgets.fixed(0.33)
                 )
def crop_model_func(year, epsilon, integration_time):
    start_date0, end_date0 = integration_time
    start_date = dt.date(year, start_date0.month, start_date0.day)
    end_date = dt.date(year, end_date0.month, end_date0.day)
    fapar = extract_smooth_fapar(year=year)
    f_water, f_temp = meteo_calculations(year=year)
    gpp = epsilon*fapar*f_water*f_temp
    plt.figure(figsize=(15, 4))
    t_axs = pd.date_range(start=dt.date(year, 1, 1),
                          end=dt.date(year, 12, 31))
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 6),
                            sharex=True,squeeze=True)
    axs = axs.flatten()
    axs[0].plot(t_axs, gpp)
    
    axs[0].axvspan(*mdates.datestr2num([start_date.strftime("%Y-%m-%d"), 
                                     end_date.strftime("%Y-%m-%d")]), color='0.9', alpha=0.5)
    doy_start = int(start_date.strftime("%j"))
    doy_end = int(end_date.strftime("%j")) + 1



    assim = gpp[doy_start:doy_end]

    axs[1].plot(pd.date_range(start=start_date, end=end_date),
             assim.cumsum())
    axs[1].set_xlim(dt.date(year, 1, 1),
             dt.date(year, 12, 31))
    axs[0].set_ylabel("GPP [funky units]")
    axs[1].set_ylabel(r'$\int GPP dt$')
    _ = axs[1].set_xlabel("Time [d]")



def plot_lai_stress():
    widgets.interact(plot_lai_stress_func,
                     year=widgets.IntSlider(min=2010, max=2018),
                     product=widgets.Dropdown(options=["LAI", "fAPAR"]))
def plot_lai_stress_func(year, product):
    product = product.lower()
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True,
                            figsize=(15, 7))
    axs = axs.flatten()
    lai = np.loadtxt(f"data/mcd15_{product}_2003_2018_-022611_106965.txt")

    df = pd.read_csv(f"data/meteo/-022611_106965/-022611_106965.{year}", skiprows=20, sep="\t", 
                     names=["station", "year", "doy", "irradiance", "tmin", "tmax", "vpd", "mws", "prec"])
    df.set_index(df['doy'])
    et0 = calc_et0(df.irradiance.values, df.tmin.values, df.tmax.values)
    f_water = water_limitation(df.prec.values, et0)
    f_temp = temp_constraint(df.tmax.values)

    axs[0].plot(df.doy, f_water, 'o', lw=0.5, mfc="none", label="Water stress")
    axs[0].plot(df.doy, f_temp, 's', lw=0.5, mfc="none", label="Heat stress")
    axs[0].plot(df.doy, np.convolve(f_water, np.ones(10)/10., mode="same"), '-', lw=3, label="Smoothed Water Stress")
    axs[0].plot(df.doy, np.convolve(f_temp, np.ones(10)/10., mode="same"), '-', lw=3, label="Smoothed Heat Stress")
    axs[0].legend(loc="best")
    axs[0].set_ylabel("Stress factor [-]")
    
    
    
    if product == "fapar":
        axs[1].plot(np.arange(1, 366, 8), lai[:, year-2003]/100, '-', lw=3)
        axs[1].set_ylabel("fAPAR [-]")
    else:
        axs[1].plot(np.arange(1, 366, 8), lai[:, year-2003]/10, '-', lw=3)
        axs[1].set_ylabel("LAI [m2/m2]")
    axs[1].set_xlabel(f"Day of year/{year} [d]")