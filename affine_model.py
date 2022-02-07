import gurobipy as grb
import numpy as np
import networkx as nx
from dm_data import dm_instance
from helper import blossom_separation
from helper import birkhoff_von_neumann_decomposition
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

    def probabilistic_alloc(self):
        # proposition 3
        if not self.optimized:
                self.optimize()
        s = copy.deepcopy(self.DM.s)
        matching = {}
        for e in self.L2G.edges:
            matching[frozenset(e)] = 0
        _residual = 0
        for xx in self.x:
            if xx[0] == 1:
                xxx = self.x[xx].X
                if abs(xxx - round(xxx)) < 1e-10:
                    xxx = round(xxx)
                matching[frozenset((xx[1],xx[2]))] = max(math.floor(xxx), 0)
                _residual += xxx % 1
        action = {}
        for c in self.Cycles:
            if frozenset(c) in matching:
                action[c] = matching[frozenset(c)]
        if _residual < 1:
            return action
        X = {}
        for e in self.L2G.edges:
            X[frozenset(e)] = 0
        for xx in self.x:
            if xx[0] == 1:
                X[frozenset((xx[1],xx[2]))] = self.x[xx].X
        G = self.L2G.copy()
        adj_list = {}
        for i in G:
            adj_list[i] = set()
        for e in G.edges:
            adj_list[e[0]].add(e[1])
            adj_list[e[1]].add(e[0])
        x_star = {}
        for i in G:
            x_star[i] = s[i] - sum([X[frozenset((i, j))] for j in adj_list[i]])
        s['D'] = math.floor(sum(x_star.values())) + 1
        x_star['D'] = s['D'] - sum(x_star.values())
        for i in x_star:
            X[frozenset(('D',i))] = x_star[i]
        G.add_node('D')
        for i in G:
            G.add_edge(i, 'D')
        adj_list['D'] = set(G.nodes)
        for i in G: 
            adj_list[i].add('D')
        xHat = {}
        for e in G.edges:
            xHat[frozenset(e)] = X[frozenset(e)] - math.floor(X[frozenset(e)])
        cHat = {}
        for i in G.nodes:
            cHat[i] = s[i] - sum([math.floor(X[frozenset((i, j))]) for j in adj_list[i]])
        vTilde = []
        for i in G.nodes:
            for k in range(1, cHat[i]+1):
                vTilde.append((i,k))
        xTilde = np.zeros((len(vTilde), len(vTilde)))
        for k in range(len(vTilde)):
            for l in range(len(vTilde)):
                i = vTilde[k][0]
                j = vTilde[l][0]
                if frozenset((i,j)) in xHat:
                    xTilde[k,l] = xHat[frozenset((i,j))] / (cHat[i] * cHat[j])
        decomp = birkhoff_von_neumann_decomposition(xTilde)
        coefs, permus = zip(*decomp)
        permu = permus[np.random.choice(range(len(coefs)), p=coefs)]
        for k in range(len(permu)):
            for l in range(len(permu)):
                if permu[k,l] == 1:
                    i = vTilde[k][0]
                    j = vTilde[l][0]
                    if i!=j:
                        e = frozenset((i, j))
                        if e in matching:
                            cHat[i] -= 1
                            cHat[j] -= 1
                            if cHat[i] >=0 and cHat[j] >=0:
                                matching[e] += 1
        for c in self.Cycles:
            if frozenset(c) in matching:
                action[c] = matching[frozenset(c)] 
        return action


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
        action = self.probabilistic_alloc()
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
        if cut == None:
            return False
        else:
            Ew = []
            bw = sum([b[i] for i in cut])
            for e in localG.edges:
                if e[0] in cut and e[1] in cut and e[0]!=e[1]:
                    Ew.append(frozenset(e))
            if sum([X[e] for e in Ew]) > ((bw-1)//2+1e-10):
                es = []
                for e in Ew:
                    e = list(e)
                    if (1, e[0], e[1]) in self.x:
                        es.append((1, e[0], e[1]))
                    if (1, e[1], e[0]) in self.x:
                        es.append((1, e[1], e[0]))
                self.r_alp.addConstr(grb.quicksum([self.x[e] for e in es]) <= (bw-1)/2)
                self.r_alp.update()
                # print('======CUT!!!======')
                return True
            else:
                return False

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
