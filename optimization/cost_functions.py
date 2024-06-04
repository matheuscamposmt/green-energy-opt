from sklearn.metrics import mean_absolute_error
import numpy as np

def cost_of_transportation(dist_matrix, forecasted_production, adj_matrix):
    return np.sum((dist_matrix * adj_matrix).T @ forecasted_production)

def cost_of_underutilization(adj_matrix, forecasted_production):
    cap_depot = 20_000
    depot_locations = np.where(adj_matrix.sum(axis=0) > 0)[0]
    return np.sum(cap_depot - (adj_matrix[:, depot_locations].T @ forecasted_production))

def cost_of_forecast(actual, forecasted):
    return mean_absolute_error(actual, forecasted)

def total_cost(dist_matrix, actual_production, forecasted_production, adj_matrix):
    a = 0.001
    b = 1
    c = 1
    return a * cost_of_transportation(dist_matrix, forecasted_production, adj_matrix) + \
           b * cost_of_underutilization(adj_matrix, forecasted_production) + \
           c * cost_of_forecast(actual_production, forecasted_production)