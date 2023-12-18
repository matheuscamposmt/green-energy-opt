from Location import Location
# Facility
class Depot(Location):
    def __init__(self, lon, lat):
        super().__init__(lon, lat)
        self.__biomass = 0

    def add_biomass(self, biomass):
        self.__biomass += biomass
    
    def get_biomass(self):
        return self.__biomass
    
    
    def __str__(self):
        return f'Facility at {self.lon, self.lat} with biomass = {self.__biomass}'
    

# test
facility = Depot(1, 2)
print(facility)
facility.add_biomass(10)
print(facility)
print(facility.get_biomass())