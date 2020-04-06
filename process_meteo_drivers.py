#!/usr/bin/env python
"""Some convenience files to grab meteorological data and convert
to CABO format to use within WOFOST. So far, using ERA5
"""
import json
import struct
import logging

import datetime as dt

from textwrap import dedent

from collections import namedtuple

from pathlib import Path

import tqdm

import numpy as np

from osgeo import gdal

from netCDF4 import Dataset, date2index

from concurrent.futures import ThreadPoolExecutor
from functools import partial


LOG = logging.getLogger(__name__)

ERAPARAMS = namedtuple(
    "ERAPARAMS", ["ssrd", "mx2t", "mn2t", "tp", "u10", "v10", "d2m"]
)


def humidity_from_dewpoint(tdew):
    """Calculates humidity from dewpoint temperature
    
    Parameters
    ----------
    tdew : float
        Dewpoint temperature in degrees Kelvin
    
    Returns
    -------
    Relative humidity
    """
    tdew = tdew - 273.15
    tmp = (17.27 * tdew) / (tdew + 237.3)
    ea = 0.6108 * np.exp(tmp)
    return ea


def retrieve_pixel_value(lon, lat, data_source):
    """Retrieve pixel value from a GDAL-friendly dataset.

    We assume the data type of the raster here!!!!
    
    Parameters
    ----------
    lon : float
        Longitude in decimal degrees
    lat : float
        Latitude in decimal degrees
    data_source : str
        An existing GDAL-readable dataset. Can be remote.
    
    Returns
    -------
    int
       The value of the pixel.
    """
    dataset = gdal.Open(data_source)

    gt = dataset.GetGeoTransform()
    the_band = dataset.GetRasterBand(1)
    px = int((lon - gt[0]) / gt[1])  # x pixel
    py = int((lat - gt[3]) / gt[5])  # y pixel

    buf = the_band.ReadRaster(px, py, 1, 1, buf_type=gdal.GDT_Int16)
    elev = struct.unpack("h", buf)

    return elev[0]


def era_to_cabo(
    site_name,
    year,
    lon,
    lat,
    elev,
    cabo_file,
    nc_file,
    parnames,
    size=0.25,
    c1=-0.18,
    c2=-0.55,
):
    """Convert ERA5 dataset to CABO format
    
    Parameters
    ----------
    site_name : str
        A site name descriptor. Can be anything, I think
    year : int
        The year
    lon : float
        The longitude in decimal degrees
    lat : float
        The latitude in decimal degrees
    cabo_file : Pathlib object
        Path where the CABO file will be written
    nc_file : Pathlib object
        Path with the netcdf file
    parnames : iter
        A list of parameters of interest
    size : float
        ERA grid size, by default 0.25
    """
    # Open netCDF file, and stuff parameters into useful
    # data structure
    ds = Dataset(str(nc_file))
    variables = (ds.variables[var][:] for var in parnames)
    pars = ERAPARAMS(*variables)
    # Check corners
    uplat = ds.variables["latitude"][:].max()
    dnlat = ds.variables["latitude"][:].min()
    uplon = ds.variables["longitude"][:].max()
    dnlon = ds.variables["longitude"][:].min()
    x = int((lon - dnlon + size / 2) / size)
    y = int((lat - uplat - size / 2) / -size)
    times = ds.variables["time"]

    # Preprocess data: calculate daily means/aggregates
    # Get the right units.
    rad = (
        np.sum(
            pars.ssrd.reshape(-1, 24, pars.ssrd.shape[1], pars.ssrd.shape[2]),
            axis=1,
        )
        / 1000.0
    )
    tmax = (
        np.max(
            pars.mx2t.reshape(-1, 24, pars.mx2t.shape[1], pars.mx2t.shape[2]),
            axis=1,
        )
        - 273.15
    )
    tmin = (
        np.min(
            pars.mn2t.reshape(-1, 24, pars.mn2t.shape[1], pars.mn2t.shape[2]),
            axis=1,
        )
        - 273.15
    )
    prec = (
        np.sum(
            pars.tp.reshape(-1, 24, pars.tp.shape[1], pars.tp.shape[2]), axis=1
        )
        * 1000.0
    )
    prec[prec < 0.01] = 0
    wind_u = np.mean(
        pars.u10.reshape(-1, 24, pars.u10.shape[1], pars.u10.shape[2]), axis=1
    )
    wind_v = np.mean(
        pars.v10.reshape(-1, 24, pars.v10.shape[1], pars.v10.shape[2]), axis=1
    )
    wind = np.sqrt(np.square(wind_u) + np.square(wind_v))
    hum = humidity_from_dewpoint(
        np.mean(
            pars.d2m.reshape(-1, 24, pars.d2m.shape[1], pars.d2m.shape[2]),
            axis=1,
        )
    )
    hdr_chunk = f"""\
        *---------------------------------------------------
        * Station: {site_name:s}
        * Year: {year:d}
        * Origin: ERA5-Reanalysis
        * Columns & units
        * ===================
        * 1. station number
        * 2. year
        * 3. Day of Year
        * 4. Irradiance   (kJ·m-2·d-1)
        * 5. Daily minimum temperature (degC)
        * 6. Daily maximum temperature (degC)
        * 7. Vapour pressure (kPa)
        * 8. Mean wind speed (m·s-1)
        * 9. Precipitation (mm·d-1)
        ** WCCDESCRIPTION={site_name:s}
        ** WCCFORMAT=2
        ** WCCYEARNR={year:d}
        *------------------------------------------------------------*
        {lon:.2f}  {lat:.2f}  {elev:.2f} {c1:.2f}  {c2:.2f}
        """
    hdr_chunk = dedent(hdr_chunk)
    # Dump data file...
    station_number = 1
    with cabo_file.open("w") as fp:
        fp.write(hdr_chunk)
        for d in range(rad.shape[0]):
            fp.write(
                f"{station_number:d}\t{year:d}\t{d+1:d}\t"
                + f"{round(rad[d,y,x]):5.1f}\t"
                + f"{round(tmin[d,y,x]*10/10):5.1f}\t"
                + f"{round(tmax[d,y,x]*10/10):5.1f}\t"
                + f"{round(hum[d,y,x]*1000/1000):5.3f}\t"
                + f"{round(wind[d,y,x]*10/10):4.1f}\t"
                + f"{round(prec[d,y,x]*10/10):4.1f}\n"
            )
    LOG.info(f"Saved CABO file {str(cabo_file):s}.")


def grab_meteo_data(
    lat,
    lon,
    year,
    era_fname,
    nc_dir,
    site_name="default_site",
    data_dir="./",
    size=0.25,
    c1=-0.18,
    c2=-0.55,
    station_number=1,
    dem_file="/vsicurl/http://www2.geog.ucl.ac.uk/"
    + "~ucfafyi/eles/global_dem.vrt",
    era_lat_chunk=1.0,
    era_lon_chunk=1.0,
):
    """Grab meteorological data and set it up to use with WOFOST.
    At present, we download the data from ERA5, but other sources may
    be considered.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    year: int
        Year of interest
    region: str
        Region name
    data_dir : str
        The location where the ERA5 file will be downloaded to, and 
        also where the CABO file will be written to
    size : float
        The grid size in degrees
    c1 : float
        I think this is a parameter related to the Armstrong exponent
    c2 : float
        I think this is a parameter related to the Armstrong exponent
    station_number : int
        A random number needed for CABO file
    dem_file : str
        A GDAL-readable file with a DEM to fish out the elevation of the site.

    Returns
    -------
        dict
        A dictionary with the relevant CABO files indexed by year.
    """
    # This is the site name. Use the longitude/latitude to make it unique
    # Grab the elevation
    elevation = retrieve_pixel_value(lon, lat, dem_file)
    # These are the parameters
    parnames = ["ssrd", "mx2t", "mn2t", "tp", "u10", "v10", "d2m"]
    return_files = {}

    cabo_file = Path(data_dir) / f"{site_name:s}.{year:d}"
    if not cabo_file.exists():
        LOG.info(f"No CABO file for {year:d}...")
        nc_file = Path(nc_dir) / (era_fname)
        if not nc_file.exists():
            LOG.info(f"No netCDF file for {year:d}...")
            raise ValueError("No NETCDF file!")
        LOG.info(f"Converting {str(nc_file):s} to CABO")
        LOG.info("Converting units to daily etc.")
        era_to_cabo(
            site_name,
            year,
            lon,
            lat,
            elevation,
            cabo_file,
            nc_file,
            parnames,
            size=size,
        )
    return cabo_file


if __name__ == "__main__":
    # Extract meteo inputs for all districts using centroid coordinates!
    districts = json.load(
        open("carto/Districts/centroid_regions.geojson", "r")
    )
    coords = [
        feat["geometry"]["coordinates"] for feat in districts["features"]
    ]
    names = [
        feat["properties"]["RGN_NM2012"] for feat in districts["features"]
    ]
    locations = dict(zip(names, coords))
    years = np.arange(2010, 2019).astype(np.int)

    for location, (lon, lat) in tqdm.tqdm(locations.items()):
        print(location, lon, lat)
        loc_name = location.replace(" ", "_")
        meteo_folder = Path(f"./data/meteo/{loc_name}")
        meteo_folder.mkdir(parents=True, exist_ok=True)
        wrapper = lambda year: grab_meteo_data(
            lat,
            lon,
            year,
            f"era5_africa_{year:d}.nc",
            "/data/geospatial_08/ucfajlg/ERA5_meteo",
            data_dir=meteo_folder.as_posix(),
            site_name=loc_name,
        )
        # create a thread pool of 10 threads

        with ThreadPoolExecutor(max_workers=2) as executor:
            for _ in executor.map(wrapper, years):
                pass
