import gurobipy as grb
from dm_data import dm_instance
import numpy as np
from affine_model import affine_model
import copy
import math
from numpy.random import RandomState
import networkx as nx


'''
    benchmark policy: LP-based randomized policy (Eq. 13) from  Ozkan and Ward (2020)
    Only compatible with Ozkan and Ward instances

'''
class ow_instance:
    def __init__(self, n):
        '''
        Ozkan and Ward (2020)
        3 areas
        3 Driver types, 3 Customer types
        r(n,j) = 1, unweighted
        f(n,j): failure probability of match (n,j)
        arr_d(n): arrival rate of driver type n [Length N list]
        arr_c(j): arrival rate of custoemr type j [Length J list]
        '''
        self.N = 3
        self.J = 3
        self.t = 10*n
        self.arr_d = [0, 0.5, 0.5]
        self.arr_c = [0.5, 0.5, 0]

        self.n = n

    def generate_instance(self):
        N, J = self.N, self.J
        self.G = nx.DiGraph()
        for n in range(N):
            self.G.add_node('D'+str(n+1))
        for j in range(J):
            self.G.add_node('C'+str(j+1))
        for n in range(N):
            for j in range(J):
                self.G.add_edge('D'+str(n+1), 'C'+ str(j+1))
                self.G.add_edge('C'+str(j+1), 'D'+ str(n+1))
        self.alpha, self.beta = {}, {}

        for n in range(N):
            self.alpha['D'+str(n+1)] = 1 - math.exp(-10*self.n)
        for j in range(J):
            self.alpha['C'+str(j+1)] = 0

        for n in range(N):
            self.beta['D'+str(n+1)] = self.arr_d[n]
        for j in range(J):
            self.beta['C'+str(j+1)] = self.arr_c[j]
        self.initState = {}
        for v in self.G.nodes:
            self.initState[v] = 0
        self.r, self.f = {}, {}
        for e in self.G.edges:
            self.r[e] = 0.5
            if e[0][0] == 'C':
                if e[0][1] == e[1][1]:
                    self.f[e] = 0
                elif eval(e[0][1]) + eval(e[1][1]) == 3:
                    self.f[e] = 0.01
                elif eval(e[0][1]) + eval(e[1][1]) == 4:
                    self.f[e] = 1
                elif eval(e[0][1]) + eval(e[1][1]) == 5:
                    self.f[e] = 0.02
            else:
                self.f[e] = 0
            
            
        self.w = {}
        for i in self.G.nodes:
            self.w[i] = 0
        a_dm = dm_instance(self.G, self.t, self.alpha, self.beta, self.initState, self.f, self.r, self.w)
        return a_dm



class ow_lp:

    def __init__(self, ow_instance, seed=42):
        self.ow_instance = copy.deepcopy(ow_instance)
        self.dm_instance = self.ow_instance.generate_instance()
        self.dm_instance.generate_cycles(2)
        self.set_drivers = list(range(1, self.ow_instance.N+1))
        self.set_customers = list(range(1, self.ow_instance.J+1))
        self.rg = RandomState(seed)
        self.F = {}
        for i in self.set_drivers:
            for j in self.set_customers:
                if ('D'+str(i), 'C'+str(j)) in self.dm_instance.mu_bar:
                    self.F[i,j] = self.dm_instance.mu_bar['D'+str(i), 'C'+str(j)]
                else:
                    self.F[i,j] = self.dm_instance.mu_bar['C'+str(j), 'D'+str(i)]

    def get_x(self):
        lp = grb.Model()
        lp.setParam('OutputFlag', 0)
        x = lp.addVars(self.set_drivers, self.set_customers)
        obj = 0
        for i in self.set_drivers:
            for j in self.set_customers:
                obj += self.ow_instance.arr_c[j-1] * self.F[i,j] * x[i,j]
        lp.setObjective(obj, grb.GRB.MAXIMIZE)

        for i in self.set_drivers:
            lp.addConstr(sum([
                    self.ow_instance.arr_c[j-1] * self.F[i,j] * x[i,j] for j in self.set_customers
                ]) <= self.ow_instance.arr_d[i-1])
        for j in self.set_customers:
            lp.addConstr(sum([
                    x[i,j] for i in self.set_drivers
                ]) <= 1)
        lp.optimize()
        self.upper_bound = lp.objVal * len(self.dm_instance.Horizon)
        self.X = {}
        for j in self.set_customers:
            prob = {}
            for i in self.set_drivers:
                if x[i,j].X > 0:
                    prob[i] = x[i,j].X
            P = sum(prob.values())
            if P < 1:
                prob['reject'] = 1-P
            self.X[j] = prob



    def randomized_lp_decision(self):
        alloc = {}
        num_customers, num_drivers = {}, {}
        for i in self.set_drivers:
            for j in self.set_customers:
                alloc[i,j] = 0
        for i in self.set_drivers:
            num_drivers[i] = self.dm_instance.s['D'+str(i)]
        for j in self.set_customers:
            num_customers[j] = self.dm_instance.s['C'+str(j)]

        local_rg = RandomState()
        for j in num_customers:
            if j == 0:
                continue
            else:
                num_j_customers = num_customers[j]
                j_choices = list(self.X[j].keys())
                j_prob = list(self.X[j].values())
                for customer in range(num_j_customers):
                    assigned_driver = local_rg.choice(j_choices, p=j_prob)
                    if assigned_driver in num_drivers and num_drivers[assigned_driver] >= 1:
                        num_drivers[assigned_driver] -= 1
                        alloc[assigned_driver, j] += 1
        reward = 0
        for i in self.set_drivers:
            for j in self.set_drivers:
                ac = self.rg.binomial(alloc[i,j], self.F[i,j])
                reward += ac
                self.dm_instance.s['D'+str(i)] -= ac
                self.dm_instance.s['C'+str(j)] -= ac
        self.dm_instance.Horizon = self.dm_instance.Horizon[:-1]
        return reward
