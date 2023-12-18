class Location:
    def __init__(self, lon, lat):
        self.lat = lat
        self.lon = lon
    
    def __str__(self):
        return f'{str(self.lon)}, {str(self.lat)}'
    
    
    