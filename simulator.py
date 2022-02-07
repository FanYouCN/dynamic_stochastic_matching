import numpy as np
import networkx as nx
from dm_data import dm_instance
from helper import blossom_separation
from helper import birkhoff_von_neumann_decomposition
from numpy.random import RandomState
import copy
from affine_model import affine_model
from alp_cg import alp_cg
from myopic import myopic_model
from limited_lookahead import limited_lookahead_model
from deterministic_lp import deterministic_lp
from instance_generator import instance_generator
from time import time
import pandas as pd


'''
    simulator class for running experiments
'''


class simulator:
    def __init__(self, aDMinstance, L):
        self.DM = copy.deepcopy(aDMinstance)
        self.L = L
        self.results = {}


    def show_result(self):
        print('-'*100)
        print(self.results)

    def get_result_pdf(self):
        self.result_pdf = pd.DataFrame.from_dict(self.results)


    def get_alp_ub(self):
        t = time()
        alp = affine_model(self.DM, self.L)
        alp.optimize()
        alp_ub = alp.r_alp.objVal
        self.results['alp_t'] = [time() - t]
        self.results['alp_ub'] = [alp_ub]


    def get_cg_ub(self):
        t = time()
        cg = alp_cg(self.DM, self.L)
        cg_ub = cg.solve_integer()
        self.results['cg_t'] = [time() - t]
        self.results['cg_ub'] = [cg_ub]


    def get_dlp_ub(self):
        t = time()
        dlp = deterministic_lp(self.DM, self.L)
        self.results['dlp_ub'] = [dlp.multi_dlp()]
        self.results['dlp_t'] = [time() - t]


    def sim_alp_primal(self, N):
        alp_primal_lb = 0
        for i in range(N):
            lb = 0
            alp = affine_model(self.DM, self.L, seed=i)
            alp.DM.rg = RandomState(i)
            while(len(alp.DM.Horizon)>0):
                if self.L == 2:
                    lb += alp.prob_solve()
                else:
                    lb += alp.solve()
                alp.DM.generate_arrival_departure()
            alp_primal_lb += lb
        self.results['alp_primal_lb'] = [alp_primal_lb/N]


    def sim_alp_dual(self, N):
        alp_dual_lb = 0
        for i in range(N):
            lb = 0
            alp = affine_model(self.DM, self.L, seed=i)
            alp.DM.rg = RandomState(i)
            while(len(alp.DM.Horizon)>0):
                lb += alp.one_step_greedy()
                alp.DM.generate_arrival_departure()
            alp_dual_lb += lb
        self.results['alp_dual_lb'] = [alp_dual_lb/N]


    def sim_myopic(self, N):
        myopic_lb = 0
        for i in range(N):
            lb = 0
            myopic_lp = myopic_model(self.DM, self.L, seed=i)
            myopic_lp.DM.rg = RandomState(i)
            while(len(myopic_lp.DM.Horizon)>0):
                lb += myopic_lp.myopic_decision()
                myopic_lp.DM.generate_arrival_departure()
            myopic_lb += lb
        self.results['myopic_lb'] = [myopic_lb/N]


    def sim_limited_lookahead(self, N):
        lla_lb = 0
        for i in range(N):
            lb = 0
            lla = limited_lookahead_model(self.DM, self.L, seed=i)
            lla.DM.rg = RandomState(i)
            while(len(lla.DM.Horizon)>0):
                lb += lla.limited_lookahead_decision()
                lla.DM.generate_arrival_departure()
            lla_lb += lb
        self.results['lla_lb'] = [lla_lb/N]


    def sim_dlp(self, N):
        dlp_lb = 0
        for i in range(N):
            lb = 0
            dlp = deterministic_lp(self.DM, self.L)
            dlp.DM.rg = RandomState(i)
            while(len(dlp.DM.Horizon)>0):
                lb += dlp.multi_dlp_solve()
                dlp.DM.generate_arrival_departure()
            dlp_lb += lb
        self.results['dlp_lb'] = [dlp_lb/N]


    def run(self, N):
        self.get_alp_ub()
        self.get_dlp_ub()
        self.sim_alp_primal(N)
        self.sim_alp_dual(N)
        self.sim_myopic(N)
        self.sim_limited_lookahead(N)
        self.sim_dlp(N)


if __name__ == '__main__':
    for i in range(100):
        gen = instance_generator(graphSize=5, graphDensity=.9, arrival=5, sojourn=15, horizon=15, B=5)
        dm = gen.generate_instance(seed=i)
        alp = affine_model(dm, 2)
        alp.prob_solve()