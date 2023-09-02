import numpy as np
import networkx as nx
from numpy.random import RandomState
import copy
from helper import simple_cycles

'''
    dynamic matching instance
    bipartite, general and long-cycle compatibility graphs
    stochastic arrival, departure, failure
    matching reward and waiting cost
'''


class dm_instance:
    def __init__(self, G, T, nu_bar, lam, s0, mu, r, w, seed=42):
        '''
        G: NetworkX DiGraph (Compatibility Graph)
        T: time periods
        s0: initial state
        '''
        self.Graph = G
        self.Nodes = G.nodes
        self.Edges = G.edges
        self.V = len(list(G.nodes))
        self.E = len(list(G.edges))
        self.r = r # edge reward
        self.w = w # node waiting cost
        self.mu = mu # edge failure rate
        self.Horizon = range(1,T+1,1)
        self.nu_bar = nu_bar # length |V| array, nu_bar_i = 1 - nu_i (departure rate)
        self.lam = lam # vetex arrival rate, length |V| array
        self.s = s0
        self.rg = RandomState(seed)
        self.seed = seed
        self.description = ''
        
    def generate_cycles(self, L):
        self.Cycles = []
        cycles = simple_cycles(self.Graph, L)
        for c in cycles:
            if len(c) == 1:
                self.Cycles.append((c[0],c[0]))
            else:
                self.Cycles.append(tuple(c))
        self.r_, self.mu_, self.mu_bar = {}, {}, {}
        self.Cycle_edges = {}
        for c in self.Cycles:
            self.Cycle_edges[c] = []
            for i in range(len(c)):
                self.Cycle_edges[c].append((c[i-1],c[i]))
        for c in self.Cycles:
            edges = self.Cycle_edges[c]
            r_c = 0
            for e in edges:
                if e in self.r:
                    r_c += self.r[e]
                else:
                    r_c += self.r[(e[1],e[0])]
            self.r_[c] = r_c
            sucesses = []
            for e in edges:
                if e in self.mu:
                    sucesses.append(1 - self.mu[e])
                else:
                    sucesses.append(1 - self.mu[(e[1],e[0])])
            self.mu_bar[c] = np.prod(sucesses)
            self.mu_[c] = 1 - self.mu_bar[c] 
        self.cycles_containing_node_i = {}
        for i in self.Nodes:
            self.cycles_containing_node_i[i] = []
        for c in self.Cycles:
            for v in c:
                self.cycles_containing_node_i[v].append(c)
        # print(sum(self.mu_.values())/len(self.mu_.values()))
        if L == 2:
            self.L2G = nx.Graph() # build undirected graph for blossom cut generation
            for i in self.Nodes:
                self.L2G.add_node(i)
            for e in self.Edges:
                self.L2G.add_edge(e[0], e[1])

    def generate_arrival_departure(self):
        ''' AFTER ACTION, generate arriving and departing pairs according to nu_bar and lambda
            Possion Arrival and Binomial Departure'''
        for i in self.Nodes:
            if self.s[i] < 0:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.s[i] = self.rg.binomial(self.s[i], self.nu_bar[i])
            self.s[i] += self.rg.poisson(self.lam[i])