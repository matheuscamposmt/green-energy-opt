from vns.Depot import Depot
from vns.HarvestingSite import HarvestingSite
import pandas as pd

distances = pd.read_csv('dataset/Distance_Matrix.csv')
class Connection:
    def __init__(self, site: HarvestingSite, facility: Depot):
        self.site = site
        self.facility = facility
        self.distance = distances.loc[site.index, facility.index] # TODO a way to get the distance between the site and the facility

    