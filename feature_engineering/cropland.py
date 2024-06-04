import rasterio as rs
import numpy as np
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool

CROPLAND_DATA = pd.read_csv('./dataset/cropland/cropland.csv')
CROPLAND_GEODATA = gpd.GeoDataFrame(CROPLAND_DATA.copy(), geometry=gpd.points_from_xy(CROPLAND_DATA['lon'], CROPLAND_DATA['lat']))


def read_file(filepath: str):
    dataset = rs.open(filepath)
    return dataset

def preprocess_rasterio(dataset, roi=None):
    """Preprocess the rasterio dataset

    Args:
        dataset (rasterio.DatasetReader): Rasterio dataset

    Returns:
        np.ndarray: Array
    """
    data = dataset.read(1)

    height = data.shape[0]
    width = data.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rs.transform.xy(dataset.transform, rows, cols)
    lons= np.array(xs)
    lats = np.array(ys)

    dataframe = pd.DataFrame({'lat': lats.flatten(), 'lon': lons.flatten(), 'cropland_frac': data.flatten()})

    if roi:
        dataframe = dataframe[(dataframe['lat'] >= roi[1]) & (dataframe['lat'] <= roi[3]) & (dataframe['lon'] >= roi[0]) & (dataframe['lon'] <= roi[2])]


    return dataframe


def get_cropland_from_polygon(polygon, cropland_data):
    """Get cropland from a polygon

    Args:
        data (pd.DataFrame): Dataframe
        polygon (gpd.GeoDataFrame): Polygon

    Returns:
        pd.DataFrame: Dataframe with cropland
    """

    block_cropland = cropland_data[cropland_data['geometry'].within(polygon)]
    cropland_frac = block_cropland['cropland_frac'].astype('float64')
    netgain = block_cropland['cropland_frac_netgain'].astype('float64')

    features = pd.Series({
        'median': cropland_frac.median(),
        'max': cropland_frac.max(),
        'min': cropland_frac.min(),
        'std': cropland_frac.std(),
        'netgain': netgain.mean(),
        
    })
        

    return features
    

def  add_cropland_feature(geodata):
    """Add cropland feature to the data

    Args:
        geodata (gpd.GeoDataFrame): Geodataframe
        cropland_file (str): Cropland file

    Returns:
        pd.DataFrame: Dataframe with cropland feature
    """
    with Pool() as pool:
        cropland_features = pool.starmap(get_cropland_from_polygon, [(block, CROPLAND_GEODATA) for block in geodata['block']])

    cropland_features = pd.DataFrame(cropland_features)
    
    geodata['median_cropland'] = cropland_features['median']
    geodata['min_cropland'] = cropland_features['min']
    geodata['min_cropland'] = cropland_features['max']
    geodata['netgain'] = cropland_features['netgain']


    return geodata

    




    
