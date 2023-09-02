import gurobipy as grb
from dm_data import dm_instance
import numpy as np
from affine_model import affine_model
import copy
import math
from numpy.random import RandomState

'''
    benchmark policy: myopic
'''

class myopic_model:
    
    def __init__(self, a_dminstance, L, seed=42):
        self.DM = copy.deepcopy(a_dminstance)
        self.L = L
        self.DM.generate_cycles(L)
        self.Cycles = self.DM.Cycles
        self.r = self.DM.r_
        self.w = self.DM.w
        self.mu = self.DM.mu_
        self.mu_bar = self.DM.mu_bar
        self.cycles_containing_node_i = self.DM.cycles_containing_node_i
        self.L2G = None
        self.rg = RandomState(seed)



    def myopic_decision(self):
        lp = grb.Model()
        lp.setParam('OutputFlag', 0)
        x = lp.addVars(self.Cycles, vtype=grb.GRB.INTEGER)
        obj = 0
        for c in self.Cycles:
            obj += self.r[c] * x[c] * self.mu_bar[c]
        lp.setObjective(obj, grb.GRB.MAXIMIZE)
        for i in self.DM.Nodes:
            lp.addConstr(sum([x[c] for c in self.cycles_containing_node_i[i]]) <= self.DM.s[i])
        lp.optimize()
        reward = 0
        for i in self.DM.Nodes:
            reward -= self.DM.s[i] * self.w[i]
        action = {}
        for c in self.Cycles:
            ac = abs(x[c].X)
            if abs(ac - round(ac)) < 1e-5:
                ac = round(ac)
            if ac % 1 > 0:
                ac = max(ac/1, 0)
            action[c] = self.rg.binomial(ac, self.mu_bar[c])
            reward += action[c] * self.r[c]
            for i in list(c):
                self.DM.s[i] -= action[c]
        self.DM.Horizon = self.DM.Horizon[:-1]
        self.optimized = False
        return reward