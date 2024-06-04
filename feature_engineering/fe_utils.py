import numpy as np
import geopandas as gpd
import pandas as pd

def points_to_polygon(lat, lon):
    """Convert a list of points to a polygon

    Args:
        lat (list): List of latitudes
        lon (list): List of longitudes

    Returns:
        shapely.geometry.Polygon: Polygon object
    """
    from shapely.geometry import Polygon
    return Polygon(zip(lon, lat))

def points_to_shapefile(lat, lon, filename='polygon.shp', dir='dataset/poly'):
    """Convert a list of points to a shapefile

    Args:
        lat (list): List of latitudes
        lon (list): List of longitudes
        filename (str, optional): Name of the shapefile. Defaults to 'polygon.shp'.
    """
    import geopandas as gpd
    import os

    if not os.path.exists(dir):
        os.mkdir(dir)
    path = dir + '/' + filename

    gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[points_to_polygon(lat, lon)])
    gdf.to_file(path)

    return path

def set_grid_blocks(df, lat_col='lat', lon_col='lon', offset=0.04):
    """Set polygon grid blocks for a given set of points

    Args:
        lat (list): List of latitudes
        lon (list): List of longitudes
        offset (float, optional): Offset for the grid blocks. Defaults to 0.04.

    Returns:
        list: List of grid blocks
    """
    from shapely.geometry import Polygon

    def set_block(lat: float, lon: float):
        return Polygon([(lon-offset, lat+offset), # top left
                        (lon+offset, lat+offset), # top right
                        (lon+offset, lat-offset), # bottom right
                        (lon-offset, lat-offset)]) # bottom left
    
    df = df.copy()
    df['block'] = df.apply(lambda row: set_block(row[lat_col], row[lon_col]), axis=1)

    # test if they're all equisized
    assert len(df.block.apply(lambda poly: poly.area).unique()) == 1

    return gpd.GeoDataFrame(df, geometry='block').set_crs('EPSG:4326')
    

def get_points_in_polygon(polygon, precision=0.001):
    """Get all points that are inside a given polygon

    Args:
        polygon (shapely.geometry.Polygon): Polygon object

    Returns:
        pd.DataFrame: Dataframe containing the lat and lon of the points
    """
    # get, with precision 0.001, all points that are inside the polygon using np.meshgrid
    # then, for each point, get the elevation using the OpenTopoData API

    # get the min and max lat and lon of the polygon
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    # create a grid of points
    lons = np.arange(min_lon, max_lon, precision)
    lats = np.arange(min_lat, max_lat, precision)
    lons, lats = np.meshgrid(lons, lats)

    points = dict(lat=lats.flatten(), lon=lons.flatten())
    return points

def add_lag(data: pd.DataFrame, n_lags=3):
    for i in range(1, n_lags + 1):
        data[f'lag{i}'] = data.groupby(['lat', 'lon'])['biomass'].shift(i)

    data.fillna(-1, inplace=True)
    return data