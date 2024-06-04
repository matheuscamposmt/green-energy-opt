import pandas as pd
import numpy as np
import geopandas as gpd
from gmalthgtparser import HgtParser
import os

class HgtFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.filename = file_path.split('\\')[-1]
        # the lower left corner coordinates are given in the name of the file
        # the N determine the degrees north latitude and the E determine the degrees east longitude
        # N19E073.hgt means 19 degrees north latitude and 73 degrees east longitude
        # use regex
        roi = self.filename.split('.')[0]
        self.lat = int(roi[1:3]) * -1 if roi[0] == 'S' else int(roi[1:3])
        self.lon = int(roi[4:7]) * -1 if roi[3] == 'W' else int(roi[4:7])

    # method for testing if a float is contained in the range of the file
    def __contains__(self, coords):
        lat, lon = coords
        return (self.lat <= lat <= self.lat + 1) and (self.lon <= lon <= self.lon + 1)

ELEVATION_HGT_FOLDER = 'dataset/elevation/downloaded_files'
HGT_FILENAMES = os.listdir(ELEVATION_HGT_FOLDER)
HGT_FILES = [HgtFile(os.path.join(ELEVATION_HGT_FOLDER, filename)) for filename in HGT_FILENAMES]

def get_points_in_polygon(polygon, precision=0.001):
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



def calculate_hgt_filename(lat, lon):
    cardinal_lat = 'N' if lat >= 0 else 'S'
    cardinal_lon = 'E' if lon >= 0 else 'W'

    name_lat = str(int(np.floor(lat)))
    name_lon = str(int(np.floor(lon))).zfill(3)

    filename = f'{cardinal_lat}{name_lat}{cardinal_lon}{name_lon}.hgt'

    return filename


def get_elevations_from_polygon(polygon):
    """Get the elevation of a given polygon from OpenTopo data

    Args:
        polygon (shapely.geometry.Polygon): Polygon object

    Returns:
        float: Elevation
    """
    points = get_points_in_polygon(polygon, precision=0.025)

    lats = points['lat']
    lons = points['lon']

    block_elevations = []
    for lat, lon in zip(lats, lons):
        filename = calculate_hgt_filename(lat, lon)
        with HgtParser(ELEVATION_HGT_FOLDER + '/'+ filename) as parser:
            coord_elevation = parser.get_elevation((lat, lon))[-1]
            
            block_elevations.append(coord_elevation)
    
    features = {
        'block_avg_elevation': np.mean(block_elevations),
        'block_std_elevation': np.std(block_elevations),
        'block_max_elevation': np.max(block_elevations)
    }


    return features

def get_elevations_from_points(lats, lons):
    """Get the elevation of a given point from OpenTopo data

    Args:
        lats (list): List of latitudes
        lons (list): List of longitudes

    Returns:
        pd.DataFrame: Dataframe with the elevation feature
    """
    elevations = []
    for lat, lon in zip(lats, lons):
        filename = calculate_hgt_filename(lat, lon)
        with HgtParser(ELEVATION_HGT_FOLDER + '/'+ filename) as parser:
            coord_elevation = parser.get_elevation((lat, lon))[-1]
            elevations.append(coord_elevation)

    return elevations
    
def add_elevation_features(geodata: gpd.GeoDataFrame):
    """Add elevation feature to the dataset

    Args:
        data (pd.DataFrame): Dataframe containing the lat and lon of the points

    Returns:
        pd.DataFrame: Dataframe with the elevation feature
    """
    from multiprocessing import Pool

    geodata = geodata.copy()

    # Create a pool of worker processes
    pool = Pool()

    # Apply multiprocessing to process the blocks
    block_features = list(pool.map(get_elevations_from_polygon, geodata['block']))

    # Close the pool of worker processes
    pool.close()
    pool.join()

    block_features = pd.DataFrame(block_features)

    data = pd.concat([geodata, block_features], axis=1)

    return data


