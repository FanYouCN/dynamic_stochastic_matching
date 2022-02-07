import gurobipy as grb
from dm_data import dm_instance
import numpy as np
from affine_model import affine_model
import copy
import math
from numpy.random import RandomState
INF = 1e6

'''
    Approximate LP solved directly by column generation
    separation subproblems solved by integer programming
'''

class alp_cg:
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


    def build_cg(self):
        self.rmp = grb.Model() 
        self.rmp.setParam('OutputFlag', 0)
        self.theta = self.rmp.addVars(self.DM.Horizon, lb=-INF)
        T_plus = len(self.DM.Horizon)+1
        self.theta[T_plus] = 0
        self.V = self.rmp.addVars(self.DM.Horizon, self.DM.Nodes, lb=-INF)
        for i in self.DM.Nodes:
            self.V[T_plus, i] = 0
        obj = self.theta[1]
        for i in self.DM.Nodes:
            obj += self.V[1,i] * self.DM.s[i]
        self.rmp.setObjective(obj, grb.GRB.MINIMIZE)
        self.cuts = set()


    def separation_integer(self, aV, atheta):
        '''
            given a solution (atheta, aV) to self.rmp, separate
        '''
        violations = []
        new_cut_found = 0
        for t in self.DM.Horizon:
            cg_ip = grb.Model()
            cg_ip.setParam('OutputFlag', 0)
            s = cg_ip.addVars(self.DM.Nodes, vtype=grb.GRB.INTEGER, ub=INF)
            x = cg_ip.addVars(self.Cycles, vtype=grb.GRB.INTEGER)
            for i in self.DM.Nodes:
                cg_ip.addConstr(sum([x[c] for c in self.cycles_containing_node_i[i]]) <= s[i])
            if t == 1:
                for i in self.DM.Nodes:
                    cg_ip.addConstr(s[i] == self.DM.s[i])
            obj = atheta[t+1] - atheta[t]
            for c in self.Cycles:
                obj += self.mu_bar[c] * (self.r[c] - sum([self.DM.nu_bar[i] * aV[t+1,i]  for i in c])) * x[c]
            for i in self.DM.Nodes:
                obj -= (aV[t,i] - self.DM.nu_bar[i] * aV[t+1,i] + self.w[i]) * s[i]
                obj += self.DM.lam[i] * aV[t+1,i]
            cg_ip.setObjective(obj, grb.GRB.MAXIMIZE)
            cg_ip.optimize()
            violation_t = cg_ip.objVal
            violations.append(violation_t) 
            if violation_t >= 1e-10:
                cut = [t]
                for i in self.DM.Nodes:
                    cut.append(int(s[i].X))
                for c in self.Cycles:
                    cut.append(int(x[c].X))
                cut = tuple(cut)
                if cut not in self.cuts:
                    new_cut_found = 1
                    self.cuts.add(cut)
                    left = self.theta[t] - self.theta[t+1]
                    for i in self.DM.Nodes:
                        left += self.V[t,i] * s[i].X 
                        left -= self.DM.nu_bar[i] * s[i].X * self.V[t+1,i]
                        left += sum([self.DM.nu_bar[i]*self.mu_bar[c]*x[c].X for c in self.cycles_containing_node_i[i]]) * self.V[t+1,i]
                        left -= self.DM.lam[i]*self.V[t+1,i]
                    right = 0
                    for c in self.Cycles:
                        right += self.mu_bar[c] * self.r[c] * x[c].X
                    for i in self.DM.Nodes:
                        right -= self.w[i] * s[i].X
                    self.rmp.addConstr(left>=right)
                    self.rmp.update()
        if new_cut_found == 0:
            return 0

        if max(violations) <= 1e-2:
            return 0
        return 1


    def solve_integer(self):
        self.build_cg()
        theta, V = {}, {}
        T_plus = len(self.DM.Horizon)+1
        for t in self.DM.Horizon:
            theta[t] = 0
            for i in self.DM.Nodes:
                V[t,i] = 0
        theta[T_plus] = 0
        for i in self.DM.Nodes:
            V[T_plus, i] = 0
        cnt = 0
        status = 1
        while status:
            cnt += 1
            status = self.separation_integer(V, theta)
            if status == 0:
                break
            else:
                self.rmp.optimize()
                for t in self.DM.Horizon:
                    theta[t] = self.theta[t].X
                    for i in self.DM.Nodes:
                        V[t,i] = self.V[t,i].X
        self.rmp.optimize()
        self.optimized = True
        return self.rmp.objVal


    def one_step_greedy(self):
        self.solve_integer()
        IP = grb.Model()
        IP.setParam('OutputFlag', 0)
        x = IP.addVars(self.Cycles, vtype=grb.GRB.INTEGER)
        obj = 0
        V = {}
        for i in self.DM.Nodes:
            if len(self.DM.Horizon) >=2:
                V[i] = self.V[2,i].X
            else:
                V[i] = 0
        for c in self.Cycles:
            obj += self.r[c] * x[c] * self.mu_bar[c]
            for i in c:
                obj -= V[i] * x[c] * self.mu_bar[c]
                obj += self.w[i] * x[c] * self.mu_bar[c]
        IP.setObjective(obj, grb.GRB.MAXIMIZE)
        for i in self.DM.Nodes:
            IP.addConstr(sum([a[c] for c in self.cycles_containing_node_i[i]]) <= self.DM.s[i])
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
        return reward


if __name__ == '__main__':
    from instance_generator import instance_generator
    from simulator import simulator
    for i in range(100):
        gen = instance_generator(graphSize=10, graphDensity=.9, arrival=5, sojourn=15, horizon=5, B=5)
        dm = gen.generate_instance(seed=i)
        sim = simulator(dm, 3)
        sim.get_alp_ub()
        sim.get_cg_ub()
        # sim.show_result()
        print(i, '*'*30)
        if sim.ubs['alp_ub']-sim.ubs['cg_ub'] > 1e-4:
            print(sim.ubs['alp_ub']-sim.ubs['cg_ub'])