import numpy as np
from scipy.spatial.distance import cdist
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# the variable neighborhood search algorithm for the p-median problem implementation from the paper "A Variable Neighborhood Search Algorithm for the p-Median Problem" of Mladenovic and Hansen (1997)

# Move evaluation procedure
# In the next procedure called Move, the change w in the objective
# function value is evaluated when the facility that is added (denoted with goin) to the current
# solution is known, while the best one to be dropped (denoted with goout) is to be found. 
# c1: the median (closest facility) of user i, i=1,...,n (cluster)
# c2: second closest facility of user i, i=1,...,n (cluster)
# d: distance (or cost) between user i and facility j, i=1,...,n, j=1,...,m
# goin: index of inserted facility (input value)
# w: change in the objective function value obtained by the best interchange
# x_curr(i), i = 1,..., p: current solution (indice of medians)
# v(j): change in the objective function value obtained by deleting each facility currently in the solution, (j= x_curr(l), l=1,...,p)
# goout: index of deleted facility (output value)
def move(c1: np.ndarray, c2: np.ndarray, d: np.ndarray, x_cur:np.ndarray, goin: int, p: int, n: int):
    w = 0
    goout = 0
    v = np.zeros(p)
    # Best deletion

    # for each user i, i=1,...,n
    for i in range(n):
        if d[i, goin] < d[i, c2[i]]:
            w = w + d[i, goin] - d[i, c1[i]]

        else:
            j = np.where(x_cur == c1[i])[0].squeeze()
            v[j] = v[j] + min(d[i, goin], d[i, c2[i]]) - d[i, c1[i]]
    
    # fing g = min{v[x_curr(l)], l=1,...,p})]} and facility goout (index x_curr(l) where this minimum is reached)
    g = np.min(v)
    goout = np.argmin(v)
    w = w + g

    return goout, w

"""This function is used to update the current solution x_curr,
the closest facility c1 and the second closest facility c2 
when the best interchange is accepted."""
def update(d, goin , goout, n, p, c1, c2):
    for i in range(n):
        if c1[i] == goout:
            if d[i, goin] <= d[i, c2[i]]:
                c1[i] = goin
            else:
                c1[i] = c2[i]
                # find second closest facility to user i
                # findf center l' where d(i, l) is minimum (for l = 1,...,p, l != c1(i))
                c2[i] = np.argmin(d[i, np.where(np.arange(p) != c1[i])])
            
        else:
            if d[i, goin] < d[i, c1[i]]:
                c2[i] = c1[i]
                c1[i] = goin
            elif d[i, goin] < d[i, c2[i]]:
                c2[i] = goin
            elif c2[i] == goout:
                # find second closest facility to user i
                # find center l' where d(i, l) is minimum (for l = 1,...,p, l != c1(i))
                c2[i] = np.argmin(d[i, np.where(np.arange(p) != c1[i])])

    return c1, c2

def fast_interchange(x, c1, c2, d, p, n):
    # get all possible medians except the ones in x
    x_all = np.arange(n) # all possible medians
    x_rest = np.setdiff1d(x_all, x) # possible medians except the ones in x
    while True:
        _w = np.inf
        for goin in x_rest:
            goout, w = move(c1, c2, d, x, goin, p, n)
            if w < _w:
                _w = w
                _goin = goin
                _goout = goout
        if _w >= 0:
            break

        # interchange position of x(_goout) with x(_goin)
        x[_goout] = _goin
        # update c1 and c2
        c1, c2 = update(d, _goin, _goout, n, p, c1, c2)

    return x, c1, c2


def vns_p_median(distances: np.ndarray, p: int, k_max: int) -> List[int]:
    n = distances.shape[0]
    # Initialization
    """Find arrays x_opt, c1 and c2 and f_opt as in initialization
    of the Interchange heuristic."""
    x_all = np.random.permutation(n)
    x_opt = x_all[:p]
    c1 = np.zeros(n, dtype=int)
    c2 = np.zeros(n, dtype=int)
    for i in range(n):
        c1[i] = np.argmin(distances[i, x_opt])
        c2[i] = np.argmin(distances[i, x_opt[np.arange(p) != c1[i]]])
    f_opt = np.sum(distances[np.arange(n), c1])
    f_cur = f_opt
    x_cur = x_opt.copy()
    c1_cur = c1.copy()
    c2_cur = c2.copy()

    # Step 1:
    k = 1
    # Step 2:
    while k < k_max:
        # Step 2.1: Shake
        x_cur, f_cur, c1_cur, c2_cur = shake(x_cur, k, c1_cur, c2_cur, f_cur, distances)

        # Step 2.2: Local search
        print(f_cur)

        x_cur, c1_cur, c2_cur = local_search(x_cur, c1_cur, c2_cur,distances, p, n)
        f_cur = np.sum(distances[np.arange(n), c1_cur])
        print(c2)

        # Step 2.3: Acceptance criterion
        if f_cur < f_opt:
            x_opt = x_cur
            f_opt = f_cur
            c1 = c1_cur
            c2 = c2_cur
            k = 1
        else:
            k += 1

    # Step 3: Return the best solution found
    return x_opt.tolist(), f_opt

def shake(x: np.ndarray, k: int, c1, c2, f_cur, d) -> np.ndarray:
    """Shake procedure: generate a new solution x_new by randomly
    removing k facilities from the current solution x."""
    x_new = x.copy()
    for j in range(1, k+1):
        # take goin at random
        goin = np.random.randint(n)
        # find goout and w
        goout, w = move(c1, c2, d, x_new, goin, p, n)
        # find c1_cur and c2_cur for such interchange using update function
        c1_cur, c2_cur = update(d, goin, goout, n, p, c1, c2)
        # update x_cur and f_cur accordingly
        x_new[goout] = goin
        f_cur = f_cur + w

    return x_new, f_cur, c1_cur, c2_cur

def local_search(x, c1, c2, d, p, n):
    # apply the interchange heuristic without the initialization step

    return fast_interchange(x, c1, c2, d, p, n)

def test_vns(n, m, p, k_max):
    # generate random data
    x = np.random.randint(0, 100, size=(n, 2))
    y = np.random.randint(0, 100, size=(m, 2))
    distances = cdist(x, y)

    # set the diagonal to 0
    distances[np.arange(n), np.arange(n)] = 0

    # run the algorithm
    solution, cost = vns_p_median(distances, p, k_max)

    # print solution
    print('Solution:')
    print(solution)

    print('Objective function value:')
    print(cost)

    # plot solution
    plt.figure(figsize=(12, 8))
    plt.scatter(x[:, 0], x[:, 1], s=1)
    plt.scatter(y[:, 0], y[:, 1], s=12, c='r')
    plt.scatter(y[solution, 0], y[solution, 1], s=12, c='g')
    # plot the connections
    #for i in range(n):
    #    plt.plot([x[i, 0], y[solution[i], 0]], [x[i, 1], y[solution[i], 1]], c='g', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

distance_matrix = pd.read_csv('dataset/Distance_Matrix.csv').drop('Unnamed: 0', axis=1)
biomass_history = pd.read_csv('dataset/Biomass_History.csv')

# Set the number of medians to be selected
p = 10

# Set the maximum number of iterations for the VNS algorithm
k_max = 8
n = 100
year = '2010'
distances = distance_matrix.to_numpy()[:n, :n]
biomass = biomass_history[year].to_numpy()[:n]

test_vns(n, n, p, k_max)


"""
solution, cost = vns_p_median(distances, p, k_max, biomass)

connections = np.argmin(distances[:, solution], axis=1)

# print solution
print('Solution:')
print(solution)
print('Objective function value:')
print(cost)

# plot solution


sns.set_style('whitegrid')
sns.set_context('paper')
sns.set(font_scale=1.1, font='Cambria')

plt.figure(figsize=(12, 8))
plt.scatter(biomass_history['Longitude'], biomass_history['Latitude'], s=1)
plt.scatter(biomass_history['Longitude'][solution], biomass_history['Latitude'][solution], s=12, c='r')
plt.annotate('p = ' + str(p), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
plt.annotate('k_max = ' + str(k_max), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=15)
plt.annotate('Cost = ' + str(cost), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=15)
# plot the connections
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""