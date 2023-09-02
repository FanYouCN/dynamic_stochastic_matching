import networkx as nx
from dm_data import dm_instance
import numpy as np
from numpy.random import RandomState
from simulator import simulator

'''
    random dynamic matching instance generator
'''


class instance_generator:
    def __init__(self, graphSize, graphDensity, arrival, sojourn, horizon, B):
        self.size = graphSize
        self.density = graphDensity
        self.arr = arrival
        self.l = sojourn
        self.t = horizon
        self.B = B


    def generate_instance(self, seed):
        rg = RandomState(seed)
        while 1:
            self.G = nx.gnp_random_graph(self.size, self.density, directed=True, seed=seed)
            if nx.is_weakly_connected(self.G):
                break
        self.initState = {}
        for i in self.G.nodes:
            self.initState[i] = rg.randint(self.B)

        self.nu_bar, self.lam = {}, {}
        for i in self.G.nodes:
            self.nu_bar[i] = 1 - np.exp(-self.l+1)
        self.vertexArrivalProb = {}
        for i in self.G.nodes:
            self.vertexArrivalProb[i] = rg.rand()
        s = sum(self.vertexArrivalProb.values())
        for i in self.G.nodes:
            self.lam[i] = self.arr * self.vertexArrivalProb[i] / s

        Edges = self.G.edges
        self.mu = {}
        self.r = {}
        for e in Edges:
            randForClassification = rg.rand()
            self.mu[e] = rg.uniform(0.0,0.2)
        for e in Edges:
            self.r[e] = rg.uniform(1,10)
        self.w = {}
        for i in self.G.nodes:
            self.w[i] = rg.uniform(0.5, 1)*0.1

        this_dm = dm_instance(self.G, self.t, self.nu_bar, self.lam, self.initState, self.mu, self.r, self.w)
        return this_dm

if __name__ == '__main__':
    gen = instance_generator(graphSize=12, graphDensity=.8, arrival=5, sojourn=15, horizon=15, B=15)
    dm = gen.generate_instance(1)
    sim = simulator(dm, 2)
    sim.run(10)



