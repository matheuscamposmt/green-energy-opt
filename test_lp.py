import numpy as np
import numpy as np

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


def test_move():
    d = np.array([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])
    p = 3
    n = 3

    x_all = np.random.permutation(n)
    x_opt = x_all[:p]
    c1 = np.zeros(n, dtype=int)
    c2 = np.zeros(n, dtype=int)
    for i in range(n):
        c1[i] = np.argmin(d[i, x_opt])
        c2[i] = np.argmin(d[i, x_opt[np.arange(p) != c1[i]]])
    x_cur = x_opt.copy()
    goin = 0

    goout, w = move(c1, c2, d, x_cur, goin, p, n)
    assert goout == 0
    assert w == 0

test_move()
