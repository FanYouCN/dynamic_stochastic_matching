import gurobipy as grb
import numpy as np
import networkx as nx
from dm_data import dm_instance
from helper import blossom_separation
from helper import prob_alloc_cg, birkhoff_von_neumann_decomposition
import copy
import math
from numpy.random import RandomState

'''
    Reduced ALP formulation, upper bound and policies
'''

class affine_model:
    def __init__(self, a_dm_instance, L, seed=2002):
        self.DM = copy.deepcopy(a_dm_instance)
        self.L = L
        self.optimized = False
        self.DM.generate_cycles(L)
        self.Cycles = self.DM.Cycles
        self.r = self.DM.r_ # cycle reward
        self.w = self.DM.w # node per period waiting cost
        self.mu = self.DM.mu_
        self.mu_bar = self.DM.mu_bar # cycle success prob
        self.cycles_containing_node_i = self.DM.cycles_containing_node_i
        self.L2G = None
        self.rg = RandomState(seed)
        self.blossom = True
        self.blossom_cuts = set()
        if L == 2:
            self.L2G = self.DM.L2G
        self.V = {}
        for i in self.DM.Nodes:
            self.V[i] = 0

    def build_alp(self):
        self.r_alp = grb.Model()
        self.r_alp.setParam('OutputFlag', 0)
        self.r_alp.setParam('Method', 2)
        self.r_alp.setParam('BarConvTol', 0.0)
        if len(self.DM.Horizon)==1:     
            self.x = self.r_alp.addVars(self.DM.Horizon, self.Cycles, vtype=grb.GRB.INTEGER, name="x")
        else:
            self.x = self.r_alp.addVars(self.DM.Horizon, self.Cycles, name="x")
        self.s = self.r_alp.addVars(self.DM.Horizon, self.DM.Nodes, name="s")
        self.constraints = {}
        obj = 0
        for t in self.DM.Horizon:
            for c in self.Cycles:
                obj += self.r[c] * self.x[(t,)+c] * self.mu_bar[c]
            for i in self.DM.Nodes:
                obj -= self.w[i] * self.s[t,i]

        self.r_alp.setObjective(obj, grb.GRB.MAXIMIZE)

        for t in self.DM.Horizon:
            for i in self.DM.Nodes:
                if t == 1:
                    self.constraints[t,i] = self.r_alp.addConstr(self.s[t,i] == self.DM.s[i])
                else:
                    self.constraints[t,i] = self.r_alp.addConstr(
                        self.s[t,i] == self.DM.nu_bar[i] * (self.s[t-1,i] - sum([self.x[(t-1,)+c]*self.mu_bar[c] for c in self.cycles_containing_node_i[i]])) + self.DM.lam[i]
                    )

        for t in self.DM.Horizon:
            for i in self.DM.Nodes:
                self.r_alp.addConstr(sum([self.x[(t,)+c] for c in self.cycles_containing_node_i[i]]) <= self.s[t,i])

        for i in self.DM.Nodes:
            if self.DM.s[i]%2 == 1:
                c = (i, i)
                if c in self.Cycles:
                    self.r_alp.addConstr(self.x[(1,)+c]<=(self.DM.s[i]-1)/2)


    def optimize(self):
        self.build_alp()
        self.r_alp.optimize()
        if self.L == 2 and self.blossom:
            while self.check_blossom():
                self.r_alp.optimize()
        self.optimized = True


    def get_V(self):
        if not self.optimized:
            self.optimize()
        for i in self.DM.Nodes:
            if len(self.DM.Horizon) >= 2:
                self.V[i] = self.constraints[2,i].Pi
            else:
                self.V[i] = 0


    def alloc(self):
        if not self.optimized:
            self.optimize()
        action = {}
        for c in self.Cycles:
            xc = self.x[(1,)+c].X
            if abs(xc - round(xc)) < 1e-10:
                xc = round(xc)
            action[c] = max(math.floor(xc), 0)
        return action

    def prob_alloc(self):
        # non-bipartite with L=2; latest
        if not self.optimized:
            self.optimize()
        G = nx.Graph()
        x0 = {}
        s = {}
        for i in self.DM.Nodes:
            s[i] = self.DM.s[i]
        action = {}
        for xx in self.x:
            if xx[0] != 1:
                continue
            elif self.x[xx].X % 1 < 1e-6:
                # only keep first period fractional cycles
                c = (xx[1], xx[2])
                a = int(self.x[xx].X)
                action[c] = a
                s[xx[1]] -= a
                s[xx[2]] -= a
                continue
            i, j = xx[1], xx[2]
            G.add_node(i)
            G.add_node(j)
            G.add_edge(i, j)
            x0[(i, j)] = self.x[xx].X
        s0 = {}
        for i in G.nodes:
            s0[i] = s[i]
        m, num_of_cg_cuts = prob_alloc_cg(G, x0, s0)
        for c in m:
            _c = (c[1], c[0]) if c not in self.Cycles else c
            action[_c] = m[c]

        return action, num_of_cg_cut


    def solve(self):
        action = self.alloc()
        reward = 0
        for i in self.DM.Nodes:
            reward -= self.DM.s[i] * self.w[i]
        for c in self.Cycles:
            ac = action[c]
            ac = self.rg.binomial(ac, self.mu_bar[c])
            reward += ac * self.r[c]
            for i in list(c):
                self.DM.s[i] -= ac
        self.DM.Horizon = self.DM.Horizon[:-1]
        self.optimized = False
        return reward


    def prob_solve(self):
        # latest and greatest prop 3
        action, num_of_cg_cuts = self.prob_alloc()
        reward = 0
        for i in self.DM.Nodes:
            reward -= self.DM.s[i] * self.w[i]
        for c in self.Cycles:
            ac = action[c]
            ac = self.rg.binomial(ac, self.mu_bar[c])
            for i in list(c):
                self.DM.s[i] -= ac
            reward += ac * self.r[c]
        self.DM.Horizon = self.DM.Horizon[:-1]
        self.optimized = False
        return reward, num_of_cg_cuts

    def check_blossom(self):
        localG = copy.deepcopy(self.L2G)
        localEdges = copy.deepcopy(localG.edges)
        for e in localEdges:
            if e[0] == e[1]:
                localG.remove_edge(e[0],e[1])
        X = {}
        for e in localG.edges:
            X[frozenset(e)] = 0
        for xx in self.x:
            if xx[0] == 1 and xx[1]!=xx[2]:
                X[frozenset((xx[1],xx[2]))] = self.x[xx].X
        b = self.DM.s
        if max(b.values()) == 0:
            return False
        cut = blossom_separation(localG, b, X)
        if cut is not None:
            cut_str = str(sorted(list(cut)))
            if cut_str in self.blossom_cuts:
                return False
            else:
                self.blossom_cuts.add(cut_str)
        if cut == None:
            return False
        else:
            Ew = []
            bw = sum([b[i] for i in cut])
            left = 0
            for c in self.Cycles:
                if c[0] in cut and c[1] in cut:
                    if c[0] == c[1]:
                        left += 2 * self.x[1, c[0], c[0]]
                    else:
                        left += self.x[1, c[0], c[1]]
            self.r_alp.addConstr(left <= (bw-1)/2)
            self.r_alp.update()
            return True

    def one_step_greedy(self):
        IP = grb.Model()
        IP.setParam('OutputFlag', 0)
        x = IP.addVars(self.Cycles, vtype=grb.GRB.INTEGER)
        obj = 0
        self.get_V()
        for c in self.Cycles:
            obj += self.r[c] * x[c] * self.mu_bar[c]
        for i in self.DM.Nodes:
            obj += self.DM.nu_bar[i] * (self.DM.s[i] - sum(
                    [self.mu_bar[c]*x[c] for c in self.cycles_containing_node_i[i]]
                )) * self.V[i]
            
        IP.setObjective(obj, grb.GRB.MAXIMIZE)
        for i in self.DM.Nodes:
            IP.addConstr(sum([x[c] for c in self.cycles_containing_node_i[i]]) <= self.DM.s[i])
        IP.optimize()
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

    def get_all_V(self):
        self.optimize()
        self.Vs = {}
        for t in self.DM.Horizon[:-1]:
            V = {}
            for i in self.DM.Nodes:
                V[i] = self.constraints[t+1,i].Pi
            self.Vs[t] = V
        V = {}
        for i in self.DM.Nodes:
            V[i] = 0
        self.Vs[self.DM.Horizon[-1]] = V

    def dual_no_resolve(self, t):
        IP = grb.Model()
        IP.setParam('OutputFlag', 0)
        x = IP.addVars(self.Cycles, vtype=grb.GRB.INTEGER)
        obj = 0
        Vt = self.Vs[t]
        for c in self.Cycles:
            obj += self.r[c] * x[c] * self.mu_bar[c]
        for i in self.DM.Nodes:
            obj += self.DM.nu_bar[i] * (self.DM.s[i] - sum(
                    [self.mu_bar[c]*x[c] for c in self.cycles_containing_node_i[i]]
                )) * Vt[i]
        IP.setObjective(obj, grb.GRB.MAXIMIZE)
        for i in self.DM.Nodes:
            IP.addConstr(sum([x[c] for c in self.cycles_containing_node_i[i]]) <= self.DM.s[i])
        IP.optimize()
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
        return reward

