import numpy as np
from collections import namedtuple


def create_random_config(N):
    """Create spin configuration of NxN lattice"""
    return 2 * np.random.randint(2, size=(N, N)) - 1

# better way:
# https://tanyaschlusser.github.io/posts/mcmc-and-the-ising-model/
def get_dH(lattice, location):
    """ H = - Sum_<ij>(s_i s_j) """
    i, j = location
    N, _ = lattice.shape
    H, Hflip = 0, 0
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ii = (i + di) % N
        jj = (j + dj) % N
        H -= lattice[ii, jj] * lattice[i, j]
        Hflip += lattice[ii, jj] * lattice[i, j]
    return Hflip - H

def step(lattice, T):
    # Pick a random position in a lattice
    (N, _) = lattice.shape
    indices = np.random.randint(0, high=N, size=2)
    i, j = indices[0], indices[1]
    
    dH = get_dH(lattice, (i, j))
    
    maybe_flip(lattice, dH, (i, j), T)
    return lattice

def maybe_flip(lattice, dH, loc, T):
    i, j = loc
    
    if dH < 0:
        lattice[i,j] = -lattice[i,j]
    elif np.random.rand() < np.exp(-dH / T):
        lattice[i, j] = -lattice[i,j]
    
    return lattice

def energy(config):
    energy = 0
    N, _ = config.shape
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S
    return energy/4.

def magnetization(config):
    return np.sum(config)
    