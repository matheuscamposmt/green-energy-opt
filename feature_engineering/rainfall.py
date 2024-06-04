import xarray as xr
import os
import numpy as np
import pandas as pd
import geopandas as gpd

DIR = './dataset/rainfall/downloaded_files'

def read_rainfall(file):
    ds = xr.open_dataset(file)
    return ds

def read_rainfall_files(files):
    datasets = []
    for file in files:
        ds = read_rainfall(file)
        datasets.append(ds)

    return datasets

def read_from_dir(dir: str):
    files = [dir+'/'+f for f in os.listdir(dir) if f.endswith('.nc4')]
    datasets = read_rainfall_files(files)

    #combine them
    combined = xr.concat(datasets, dim='time')
    return combined

def convert_to_pd_dataframe(ds):
    df = ds.to_dataframe()

    return df

def get_rainfall_from_polygon(polygon, rainfall_df):
    """Get the rainfall of a given polygon

    Args:
        polygon (shapely.geometry.Polygon): Polygon object

    Returns:
        float: Rainfall
    """
    # get all points in the rainfall_df that are within the polygon
    block_rainfall = rainfall_df[rainfall_df['geometry'].within(polygon)]

    avg_precip_rate = block_rainfall['avg_precipitation_rate']
    max_precip_rate = block_rainfall['max_precipitation_rate']


    features = pd.Series({
        'avg_precipitation_rate': avg_precip_rate.values[0],
        'max_precipitation_rate': max_precip_rate.values[0]

    })

    return features


def preprocess_rainfall_data(data):
    data = data.to_dataframe().reset_index()
    data['year'] = data['time'].apply(lambda datetime: datetime.year)
    data = data.groupby(['year', 'lat', 'lon']).agg(avg_precipitation_rate=('precipitation', 'mean'),
                                                        max_precipitation_rate=('precipitation', 'max')
                                                        ).reset_index()
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['lon'], data['lat'])).set_crs('EPSG:4326')
    return data


def add_rainfall_feature(data, forecast_horizon: int, years=[]):
    geodata = gpd.GeoDataFrame(data.copy(), geometry='block')

    rainfall_df = read_from_dir(DIR)
    rainfall_df = preprocess_rainfall_data(rainfall_df)
    
    for year in years:
        geodata_year = geodata.query(f'year == {year}')
        rainfall_data_year = rainfall_df.query(f'year == {int(year) - forecast_horizon}')

        if rainfall_data_year.empty:
            geodata.loc[geodata_year.index, 'avg_precipitation_rate'] = -1
            continue

        geodata_year = geodata_year['block'].apply(lambda block: get_rainfall_from_polygon(block, rainfall_data_year))
        geodata.loc[geodata_year.index, 'avg_precipitation_rate'] = geodata_year['avg_precipitation_rate']
        #geodata.loc[geodata_year.index, 'max_precipitation_rate'] = geodata_year['max_precipitation_rate']

    annual_precipitation_rate = geodata.groupby('year')['avg_precipitation_rate'].sum()
    
    geodata['annual_precip_rate'] = geodata['year'].map(annual_precipitation_rate)

    return geodata
