import pandas as pd
import numpy as np

distance_matrix = pd.read_csv('dataset/Distance_Matrix.csv')
biomass_history = pd.read_csv('dataset/Biomass_History.csv')

def distance(i, j):
    return distance_matrix.iloc[i, j]



# comentários
"""Certamente! Vamos modelar o problema de otimização de seleção das localizações 
dos depósitos e refinarias como um problema de programação linear inteira (Linear Integer Programming - LIP) 
usando a biblioteca CVXPY. O objetivo é minimizar os custos de transporte e a incompatibilidade da previsão de biomassa, 
conforme descrito no documento.
"""


### Implementação:
"""
1. *Variáveis de Decisão*:
   - `x[i,j]`: Variável binária que indica se o depósito `j` é abastecido pelo local de colheita `i`.
   - `y[j,k]`: Variável binária que indica se a refinaria `k` é abastecida pelo depósito `j`.
   - `depot[j]`: Variável binária que indica se o depósito `j` é selecionado.
   - `refinery[k]`: Variável binária que indica se a refinaria `k` é selecionada.

2. *Função Objetivo*:
   - Minimizar o custo total de transporte e a incompatibilidade da previsão de biomassa.

3. *Restrições*:
   - Cada local de colheita deve ser atribuído a exatamente um depósito.
   - Cada depósito deve ser atribuído a exatamente uma refinaria.
   - Restrições de capacidade para depósitos e refinarias.
   - Restrições adicionais conforme o problema.

"""
# Define the decision variables

num_warehouses = 5
num_harvesting_sites = 25
num_possible_warehouses_locations = num_harvesting_sites

D = np.random.randint(1, 100, size=(num_harvesting_sites, num_harvesting_sites))
np.fill_diagonal(D, 0)
D = (D + D.T) / 2

B = np.random.randint(100, 1000, size=num_harvesting_sites)
C = 1000

def print_matrix(matrix):
    print(pd.DataFrame(matrix).to_string(index=True, header=True))

print_matrix(D)
import pulp

warehouse_locations = [i for i in range(num_possible_warehouses_locations)]  # Creates a list of all warehouses
# Creates a list of all demand nodes
harvest_sites = [i for i in range(num_harvesting_sites)]

cost_matrix = D * B
# The cost data is made into a dictionary
costs = pulp.makeDict([harvest_sites, warehouse_locations], cost_matrix, 0)
warehouses = pulp.LpVariable.dicts("W%s", warehouse_locations, cat=pulp.LpBinary)
serv_harvest=pulp.LpVariable.dicts("H%s_W%s", (harvest_sites, warehouse_locations), cat=pulp.LpBinary)
# Creates the 'prob' variable to contain the problem data
prob = pulp.LpProblem("Supply_Chain_Optimization", pulp.LpMinimize)

cost_of_transportation = pulp.lpSum([serv_harvest[h][w] * costs[h][w] for w in warehouse_locations for h in harvest_sites])
#cost_of_underutilization = pulp.lpSum([C - pulp.lpSum([variables[w][h] * (B[int(h)-1]) for h in Harvest_sites]) for w in Warehouses_locations])
prob += cost_of_transportation #+ cost_of_underutilization
# Constraints

# Constraint 1: Number of warehouses must be less or equal to 5
prob += (
    pulp.lpSum([warehouses[w] for w in warehouse_locations]) <= num_warehouses, 
    "Maximum_number_of_warehouses"
    )

# Constraint 2: Each harvesting site must be assigned to exactly one warehouse
for h in harvest_sites:
    prob += (
        pulp.lpSum([serv_harvest[h][w] for w in warehouse_locations]) == 1,
        f"Harvesting_site_{h}_must_be_assigned_to_exactly_one_warehouse"
    )

# Constraint 3: If a harvesting site is assigned to a warehouse, the warehouse must be selected
for h in harvest_sites:
    for w in warehouse_locations:
        prob += (
            serv_harvest[h][w] <= warehouses[w],
            f"Harvesting_site_{h}_must_be_assigned_to_warehouse_{w}_only_if_warehouse_{w}_is_selected"
        )

prob.solve()
# print the status of the solution
print("Status:", pulp.LpStatus[prob.status])

# print the value of the objective
print("Total Cost of Transportation = ", pulp.value(prob.objective))

# print the value of the variables
for v in prob.variables():
    print(v.name, "=", v.varValue)

# plot the solution on a map
import plotly.express as px

# create a dataframe with the solution
solution = pd.DataFrame(columns=['Harvesting Site', 'Warehouse', 'Selected'])
for v in prob.variables():
    if v.name.startswith('H'):
        splitted = v.name.split('_')

        solution = pd.concat(
            [solution, pd.DataFrame([[splitted[0][1:], splitted[1][1:], v.varValue]], columns=['Harvesting Site', 'Warehouse', 'Selected'])]
            )
solution = solution[solution['Selected'] == 1].reset_index(drop=True)

solution['HS_Longitude']= solution['Harvesting Site'].apply(lambda idx: biomass_history['Longitude'][int(idx)])
solution['HS_Latitude']= solution['Harvesting Site'].apply(lambda idx: biomass_history['Latitude'][int(idx)])
solution['W_Longitude']= solution['Warehouse'].apply(lambda idx: biomass_history['Longitude'][int(idx)])
solution['W_Latitude']= solution['Warehouse'].apply(lambda idx: biomass_history['Latitude'][int(idx)])

# graph representation



