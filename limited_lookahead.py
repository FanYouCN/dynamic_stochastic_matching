import gurobipy as grb
from dm_data import dm_instance
import numpy as np
from affine_model import affine_model
import copy
import math
from numpy.random import RandomState

'''
    benchmark policy: limited lookahead
'''

class limited_lookahead_model:
    def __init__(self, aDMinstance, L, seed=42):
        self.DM = copy.deepcopy(aDMinstance)
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


    def limited_lookahead_decision(self, T=5):
        lookahead_dlp = grb.Model()
        lookahead_dlp.setParam('OutputFlag', 0)
        if T > len(self.DM.Horizon):
            T = len(self.DM.Horizon)
        lookahead_horizon = range(1,T+1,1)
        x = lookahead_dlp.addVars(lookahead_horizon, self.Cycles)
        s = lookahead_dlp.addVars(lookahead_horizon, self.DM.Nodes)
        obj = 0
        for t in lookahead_horizon:
            for i in self.DM.Nodes:
                obj -= self.w[i] * s[t,i]
            for c in self.Cycles:
                obj += self.r[c] * x[(t,)+c] * self.mu_bar[c]
        lookahead_dlp.setObjective(obj, grb.GRB.MAXIMIZE)
        for t in lookahead_horizon:
            for i in self.DM.Nodes:
                if t == 1:
                    lookahead_dlp.addConstr(s[t,i] == self.DM.s[i])
                else:
                    lookahead_dlp.addConstr(s[t,i] == self.DM.nu_bar[i] * (s[t-1,i] - sum([x[(t-1,)+c]*self.mu_bar[c] for c in self.cycles_containing_node_i[i]])) + self.DM.lam[i])
        for t in lookahead_horizon:
            for i in self.DM.Nodes:
                lookahead_dlp.addConstr(sum([x[(t,)+c] for c in self.cycles_containing_node_i[i]]) <= s[t,i])
        lookahead_dlp.optimize()
        reward = 0
        for i in self.DM.Nodes:
            reward -= self.DM.s[i] * self.w[i]
        action = {}
        for c in self.Cycles:
            ac = abs(x[(1,)+c].X)
            if abs(ac - round(ac)) < 1e-5:
                ac = round(ac)
            if ac % 1 > 0:
                ac = max(ac/1, 0)
            action[c] = self.rg.binomial(ac, self.mu_bar[c])
            reward += action[c] * self.r[c]
            for i in list(c):
                self.DM.s[i] -= action[c]
        self.DM.Horizon = self.DM.Horizon[:-1]
        return reward