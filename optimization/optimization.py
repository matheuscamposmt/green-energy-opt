import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from .cost_functions import total_cost

BIOMASS = pd.read_csv('dataset/Biomass_History.csv')

def gaussian_2d(n, depot_locations, k):
    # Assuming 'biomass' and 'best_adj_matrix' are already defined
    # Extract depot coordinates
    depot_coords = BIOMASS.iloc[depot_locations][['lat', 'lon']]

    # Step 1: Calculate the mean location of the existing depots
    mean_location = depot_coords.mean()

    # Step 2: Assign probabilities to existing depots using a 2D Gaussian function
    mean_vector = mean_location.values
    std = depot_coords.std() * (k / 2)
    cov_matrix = np.diag([std['lat'], std['lon']])  # Covariance matrix based on existing depot locations
    # increase the latitudinal std deviation
    cov_matrix[0, 0] *= 4
    cov_matrix[1, 1] *= 2


    rv = multivariate_normal(mean=mean_vector, cov=cov_matrix)
    probabilities = rv.pdf(BIOMASS[['lat', 'lon']].values)  # Calculate probabilities for all biomass locations
    probabilities /= np.sum(probabilities)  # Normalize to sum to 1

    # Step 3: Sample indices of existing depots from the multinomial distribution
    num_new_depots = n
    samples = np.random.multinomial(num_new_depots, probabilities)

    new_depot_indices = np.repeat(np.arange(2418), samples)

    # if has repeated values, sample again
    while len(new_depot_indices) != len(set(new_depot_indices)):
        samples = np.random.multinomial(num_new_depots, probabilities)
        new_depot_indices = np.repeat(np.arange(2418), samples)

    return new_depot_indices

def get_site_locations(y_forecast):
    return y_forecast.argsort()[-2418:][::-1]

def get_depot_locations(y_forecast):
    return y_forecast.argsort()[-1000:][::-1]

def best_depot_locations(dist, y_forecast, n):

    site_locations = get_site_locations(y_forecast)
    depot_locations = get_depot_locations(y_forecast)

    adj_matrix = np.zeros((2418, 2418))
    index_array = np.ix_(site_locations, depot_locations)
    adj_matrix[index_array] = 1

    depots_total_biomass = (dist * adj_matrix).T @ y_forecast

    depots_index_sorted = depots_total_biomass.argsort()
    
    #removes zero
    depots_index_sorted = depots_index_sorted[depots_total_biomass[depots_index_sorted] > 0]

    n = min(n, len(depots_index_sorted))

    return depots_index_sorted[:n]


def best_site_locations(dist, y_forecast, depot_index):
    locations = get_site_locations(y_forecast)
    
    locations = locations[locations != depot_index]
    index_sorted = np.argsort((dist* y_forecast.reshape(-1, 1))[locations, depot_index])

    return locations[index_sorted]

import numpy as np

def assign_sites_to_depots(depot_locations, dist_matrix, y_forecast):
    adj_matrix = np.zeros((2418, 2418))
    depot_capacities = {depot: 0 for depot in depot_locations}
    
    for depot_location in depot_locations:
        site_locations_sorted = best_site_locations(dist_matrix, y_forecast, depot_location)
        total_depot_biomass = 0

        for site_index in site_locations_sorted:
            if np.any(adj_matrix[site_index] > 0):  # Skip if the site is already assigned
                continue
            site_biomass = y_forecast[site_index]
            # if the depot has enough capacity
            if total_depot_biomass + site_biomass <= 20_000:
                # assign the site to the depot
                adj_matrix[site_index, depot_location] = 1
                # update the total biomass of the depot
                total_depot_biomass += site_biomass
                # update the depot capacity
                depot_capacities[depot_location] += site_biomass
            # if the depot has no capacity left
            else:
                # calculate the remaining capacity
                remaining_capacity = 20_000 - total_depot_biomass
                # if the remaining capacity is greater than the site biomass
                if site_biomass <= remaining_capacity:
                    # assign the site to the depot
                    adj_matrix[site_index, depot_location] = 1
                    # update the total biomass of the depot
                    total_depot_biomass += site_biomass
                    # update the depot capacity
                    depot_capacities[depot_location] += site_biomass

            if total_depot_biomass >= 20_000:
                break
    
    return adj_matrix


class SimulatedAnnealing:
    def __init__(self, initial_temp, cooling_rate, max_iterations):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.dist_matrix = None
        self.y_forecast = None
        self.y_actual = None
        self.k = None

    def objective_function(self, adj_matrix):
        return total_cost(self.dist_matrix, self.y_actual, self.y_forecast, adj_matrix)
    
    def random_initial_solution(self):
        greedy_depot_locations = best_depot_locations(self.dist_matrix, self.y_forecast, 15)
        depot_locations = gaussian_2d(15, greedy_depot_locations, k= 2)
        
        return (assign_sites_to_depots(depot_locations, self.dist_matrix, self.y_forecast), depot_locations)
    
    def _get_depot_location(self, site_location, adj_matrix):
        return np.where(adj_matrix[site_location, :] > 0)[0].squeeze()
    
    def _get_sites_assigned_to_depot(self, depot_location, adj_matrix):
        return np.where(adj_matrix[:, depot_location] > 0)[0]
    
    def _get_depot_capacity(self, depot_location, adj_matrix):
        # get the biomass of the sites assigned to the depot
        return np.sum(adj_matrix[:, depot_location] * self.y_forecast)
    
    def _get_random_location(self, depot_locations):
        return gaussian_2d(1, depot_locations, self.k)[0]

    def elect_new_depot(self, adj_matrix, depot_locations):
        # Select a random depot to remove
        depot_to_remove = depot_locations[np.random.randint(0, depot_locations.size - 1)]

        # Select a random location
        new_depot_location = self._get_random_location(depot_locations)
        
        # if the new_depots location is already a depot
        if new_depot_location in depot_locations:
            # do nothing
            return adj_matrix
        
        # Find the sites assigned to the depot
        sites_assigned_to_depot = self._get_sites_assigned_to_depot(depot_to_remove, adj_matrix)
        depot_locations[np.where(depot_locations == depot_to_remove)] = new_depot_location

        # assign the sites to the new depot
        adj_matrix[sites_assigned_to_depot, depot_to_remove] = 0
        adj_matrix[sites_assigned_to_depot, new_depot_location] = 1


    def perturb_solution(self, current_solution):
        new_adj_matrix = current_solution[0].copy()
        new_depot_locations = current_solution[1].copy()
        if self.k > 3:
            # elect a new depot
            self.elect_new_depot(new_adj_matrix, new_depot_locations)
            
        # Select a random site
        random_site = np.random.randint(0, self.y_forecast.shape[0] - 1)
        site_biomass = self.y_forecast[random_site]
        # Find the depot currently assigned to the selected site
        current_depot = self._get_depot_location(random_site, new_adj_matrix)
        # Find the depots locations with biomass capacity
        depots_with_capacity = new_depot_locations[np.where(new_adj_matrix[:, new_depot_locations].T @ self.y_forecast + site_biomass <= 20_000)[0]]

        # if there are depots with capacity
        if depots_with_capacity.size > 0:

            random_depot = depots_with_capacity[np.random.randint(0, depots_with_capacity.size - 1)]

            # if the site has a depot assigned, swap the site to a new depot
            if current_depot.size > 0:
                current_depot_capacity = self._get_depot_capacity(current_depot, new_adj_matrix)

                #print(f"\t Site {random_site}(biomass: {site_biomass}) is currently assigned to depot {current_depot} ({current_depot_capacity})")
                
                random_depot_capacity = self._get_depot_capacity(random_depot, new_adj_matrix)

                # assign the site to the new depot
                new_adj_matrix[random_site, current_depot] = 0
                new_adj_matrix[random_site, random_depot] = 1


                
            # if doesn't have a depot assigned, assign it to a depot
            else:
                # assign the site to the new depot
                new_adj_matrix[random_site, random_depot] = 1

        new_solution = (new_adj_matrix, new_depot_locations)

        return new_solution

    
    def acceptance_probability(self, old_cost, new_cost, temperature):

        if new_cost < old_cost:
            return 1.0
        return np.exp((old_cost - new_cost) / temperature)
    
    def fit(self, dist_matrix, y_forecast, y_actual):
        self.dist_matrix = dist_matrix
        self.y_forecast = y_forecast
        self.y_actual = y_actual

        current_solution = self.random_initial_solution()
        current_cost = self.objective_function(current_solution[0])
        best_solution = current_solution
        best_cost = current_cost
        temperature = self.initial_temp

        print("Initial Solution Cost:\n", current_cost)

        self.k = 0
        for iteration in range(self.max_iterations):
            #print(f"[{iteration}] iteration:")
            new_solution = self.perturb_solution(current_solution)
            new_cost = self.objective_function(new_solution[0])
            
            # Accept the new solution if it's better or with a certain probability if it's worse
            if self.acceptance_probability(current_cost, new_cost, temperature) > np.random.random():
                current_solution = new_solution
                current_cost = new_cost
                
                # Update the best solution found
                if new_cost < best_cost:    
                    best_solution = new_solution
                    best_cost = new_cost
                    print("New Best Solution Found! Cost:", best_cost)
                    self.k = 0

                else:
                    self.k += 0.5


            temperature *= self.cooling_rate

            if iteration % 100 == 0:
                print(f"Iteration: {iteration}, Temperature: {temperature:.6f}, Current Cost: {current_cost}, Best Cost: {best_cost}, k: {self.k}")

            if temperature < 1e-10:  # Stopping criterion
                break

        print("Final Best Solution Cost:", best_cost)

        return best_solution, best_cost