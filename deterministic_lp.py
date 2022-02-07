import gurobipy as grb
from dm_data import dm_instance
import numpy as np
from affine_model import affine_model
import copy
import math
from numpy.random import RandomState

'''
    deterministic linear programming formulation
    system dynamics replaced by expected values
    upper bound and control policy
'''


class deterministic_lp:
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
        self.V = {}
        for i in  self.DM.Nodes:
            self.V[i] = 0


    def multi_dlp(self):
        dlp = grb.Model()
        dlp.setParam('OutputFlag', 0)
        x = dlp.addVars(self.DM.Horizon, self.Cycles)
        s = dlp.addVars(self.DM.Horizon, self.DM.Nodes)
        obj = 0
        for t in self.DM.Horizon:
            for i in self.DM.Nodes:
                obj -= self.w[i] * s[t,i]
            for c in self.Cycles:
                obj += self.r[c] * x[(t,)+c] * self.mu_bar[c]
        dlp.setObjective(obj, grb.GRB.MAXIMIZE)
        constraints = {}
        for t in self.DM.Horizon:
            for i in self.DM.Nodes:
                if t == 1:
                    dlp.addConstr(s[t,i] == self.DM.s[i])
                else:
                    constraints[t, i] = dlp.addConstr(s[t,i] == self.DM.nu_bar[i] * (s[t-1,i] - sum([x[(t-1,)+c]*self.mu_bar[c] for c in self.cycles_containing_node_i[i]])) + self.DM.lam[i])
        T = self.DM.Horizon[-1]
        for i in self.DM.Nodes:
            dlp.addConstr(self.DM.nu_bar[i] * (s[T,i] - sum([x[(T,)+c]*self.mu_bar[c] for c in self.cycles_containing_node_i[i]])) + self.DM.lam[i]>=0)
        dlp.optimize()
        if T > 1:
            for i in self.DM.Nodes:
                self.V[i] = constraints[2,i].Pi 
        return dlp.objVal


    def multi_dlp_solve(self):
        self.multi_dlp()
        IP = grb.Model()
        IP.setParam('OutputFlag', 0)
        x = IP.addVars(self.Cycles, vtype=grb.GRB.INTEGER)
        obj = 0
        for c in self.Cycles:
            obj += self.r[c] * x[c] * self.mu_bar[c] 

        IP.setObjective(obj, grb.GRB.MAXIMIZE)
        for i in self.DM.Nodes:
            IP.addConstr(sum([x[c] for c in self.cycles_containing_node_i[i]]) <= self.DM.s[i])
        for c in self.Cycles:
            if self.r[c]+self.mu[c]*(sum([self.V[i] for i in c])) - sum([self.DM.nu_bar[i]*self.V[i] for i in c]) <= 0:
                IP.addConstr(x[c] == 0)
        IP.optimize()
        reward = 0
        for i in self.DM.Nodes:
            reward -= self.DM.s[i] * self.w[i]
        action = {}
        for c in self.Cycles:
            ac = round(x[c].X)
            action[c] = self.rg.binomial(ac, self.mu_bar[c])
            reward += action[c] * self.r[c]
            for i in list(c):
                self.DM.s[i] -= action[c]
        self.DM.Horizon = self.DM.Horizon[:-1]
        return reward
