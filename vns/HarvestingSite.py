from Location import Location

class HarvestingSite(Location):
    def __init__(self, lon, lat, biomass):
        super().__init__(lon, lat)
        self.__biomass = biomass
    
    def get_biomass(self):
        return self.__biomass
    
    def __str__(self):
        return f'Harvesting Site at {self.lon, self.lat} with biomass = {self.__biomass}'