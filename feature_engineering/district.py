import pandas as pd
import geopandas as gpd


def get_districts_from_shp(shapefile: str):
    """Get district names from shapefile

    Args:
        shapefile (str): Path to the shapefile

    Returns:
        list: List of district names
    """
    gdf = gpd.read_file(shapefile)
    
    return gdf

DISTRICTS = get_districts_from_shp('dataset/districts/india.shp').to_crs('EPSG:4326')

def add_district_feature_block(geodata: gpd.GeoDataFrame):
    """Add district feature to the dataset

    Args:
        data (pd.DataFrame): Dataframe containing the lat and longitude of the points

    Returns:
        pd.DataFrame: Dataframe with the district feature
    """
    geodata = geodata.copy()

    overlay = gpd.overlay(geodata, DISTRICTS, how='intersection')
    overlay['area'] = overlay['geometry'].apply(lambda x: x.area)
    overlay = overlay.sort_values(by='area')

    overlay = overlay.drop_duplicates(subset='index', keep='last')

    # perform a spatial join between the geodata and districts polygons (lat, lon and geometry)
    data = geodata.merge(overlay[['index', 'distname']], on='index', how='left')

    data['distname'] = data['distname'].fillna('Sea')

    return data

def add_district_feature(data: pd.DataFrame):
    """Add district feature to the dataset

    Args:
        data (pd.DataFrame): Dataframe containing the lat and longitude of the points

    Returns:
        pd.DataFrame: Dataframe with the district feature
    """
    data = data.copy()
    geodata_from_df = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat)).set_crs('EPSG:4326')

    joined_data = gpd.sjoin(geodata_from_df, DISTRICTS, how='left', predicate='within')
    data['distname'] = joined_data['District'].str.replace('>', 'A')
    data['distname'] = data['distname'].str.lower()

    return data
