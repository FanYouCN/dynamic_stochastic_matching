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
from ow_lp import ow_lp
from deterministic_lp import deterministic_lp
from time import time
import pandas as pd
from scipy.stats import sem
from prob_cg import prob_alloc_cg_alt
import os

    
'''
    simulator class for running experiments
'''

TIME_LIMIT = 600

class simulator:
    def __init__(self, a_dminstance, L):
        self.DM = copy.deepcopy(a_dminstance)
        self.L = L
        self.helper_dict = {
            'sample_path_index': [],
            'field': [],
            'value': []
        }
        self.output_path = 'res/' + self.DM.description + '.csv'
        _df = pd.DataFrame.from_dict(self.helper_dict)
        _df.to_csv(self.output_path, index=False, mode='w')
        self.results = {}
        self.seed = None

    def write_row(self, sample_path_index, field, value):
        self.helper_dict['sample_path_index'] = [sample_path_index]
        self.helper_dict['field'] = [field]
        self.helper_dict['value'] = [value]
        _df = pd.DataFrame.from_dict(self.helper_dict)
        _df.to_csv(self.output_path, index=False, mode='a', header=False)

    def get_alp_ub(self):
        t = time()
        alp = affine_model(self.DM, self.L)
        alp.optimize()
        alp_ub = alp.r_alp.objVal
        self.write_row(-1, 'alp_t', time() - t)
        self.write_row(-1, 'alp_ub', alp_ub)
        self.results['alp_ub'] = alp_ub

    def get_cg_ub(self, verbose=0):
        t = time()
        cg = alp_cg(self.DM, self.L)
        cg_ub = cg.solve_integer(verbose)
        self.write_row(-1, 'cg_t', time() - t)
        self.write_row(-1, 'cg_ub', cg_ub)
        self.results['cg_ub'] = cg_ub

    def get_dlp_ub(self):
        dlp = deterministic_lp(self.DM, self.L)
        t = time()
        dlp_ub = dlp.multi_dlp()
        self.write_row(-1, 'dlp_t', time() - t)
        self.write_row(-1, 'dlp_ub', dlp_ub)
        self.results['dlp_ub'] = dlp_ub

    def sim_alp_dual(self, N, seed=0):
        # latest and greatest prop 3
        lbs = []
        num_of_cg_cuts_list = []
        for i in range(N):
            # print(self.DM.description, 'ALP Dual sample path: ', i)
            t0 = time()
            lb = 0
            alp = affine_model(self.DM, self.L, seed=i+seed)
            alp.DM.rg = RandomState(i+seed)
            while(len(alp.DM.Horizon)>0):
                print('Horizon: ', len(alp.DM.Horizon), '     ', end=' \r')
                if time() - t0 > TIME_LIMIT:
                    lb = -1
                    break
                if self.L == 2:
                    reward, num_of_cg_cuts = alp.prob_solve()
                    lb += reward
                    num_of_cg_cuts_list.append(num_of_cg_cuts)
                else:
                    lb += alp.solve()
                alp.DM.generate_arrival_departure()
            lbs.append(lb)
            self.write_row(i, 'alp_dual_lb', lb)
            if self.L == 2:
                self.write_row(i, 'avg_alp_dual_cg_cuts', sum(num_of_cg_cuts_list) / len(num_of_cg_cuts_list))
        return sum(lbs) / N



    def sim_alp_primal(self, N, seed=0):
        lbs = []
        for i in range(N):
            print(self.DM.description, 'ALP Primal sample path: ', i)
            t0 = time()
            lb = 0
            alp = affine_model(self.DM, self.L, seed=i+seed)
            alp.DM.rg = RandomState(i+seed)
            while(len(alp.DM.Horizon)>0):
                print(i, 'Horizon: ', len(alp.DM.Horizon), '  ', end='\r')
                if time() - t0 > TIME_LIMIT:
                    lb = -1
                    break
                lb += alp.one_step_greedy()
                alp.DM.generate_arrival_departure()
            lbs.append(lb)
            self.write_row(i, 'alp_primal_lb', lb)


    def sim_alp_primal_no_re(self, N, seed=0):
        lbs = []
        for i in range(N):
            print(self.DM.description, 'ALP Primal no-re sample path: ', i)
            t0 = time()
            lb = 0
            alp = affine_model(self.DM, self.L, seed=i+seed)
            alp.get_all_V()
            alp.DM.rg = RandomState(i+seed)
            for t in self.DM.Horizon:
                if time() - t0 > TIME_LIMIT:
                    lb = -1
                    break
                lb += alp.dual_no_resolve(t)
                alp.DM.generate_arrival_departure()
            lbs.append(lb)
            self.write_row(i, 'alp_primal_lb_no_re', lb)

    def sim_myopic(self, N, seed=0):
        lbs = []
        for i in range(N):
            print(self.DM.description, 'Myopic Primal sample path: ', i)
            t0 = time()
            lb = 0
            myopic_lp = myopic_model(self.DM, self.L, seed=i+seed)
            myopic_lp.DM.rg = RandomState(i+seed)
            while(len(myopic_lp.DM.Horizon)>0):
                print('Horizon: ', len(alp.DM.Horizon), '  ', end='\r')
                if time() - t0 > TIME_LIMIT:
                    lb = -1
                    break
                lb += myopic_lp.myopic_decision()
                myopic_lp.DM.generate_arrival_departure()
            lbs.append(lb)
            self.write_row(i, 'myopoc_lb', lb)


    def sim_limited_lookahead(self, N, seed=0):
        if 'alp_ub' not in self.results:
            self.get_alp_ub()
        lbs = []
        for i in range(N):
            print(self.DM.description, 'LLA sample path: ', i)
            t0 = time()
            lb = 0
            lla = limited_lookahead_model(self.DM, self.L, seed=i+seed)
            lla.DM.rg = RandomState(i+seed)
            while(len(lla.DM.Horizon)>0):
                print('Horizon: ', len(lla.DM.Horizon), '  ', end='\r')
                if time() - t0 > TIME_LIMIT:
                    lb = -1
                    break
                lb += lla.limited_lookahead_decision()
                lla.DM.generate_arrival_departure()
            lbs.append(lb)
            self.write_row(i, 'lla_lb', lb)

    def sim_dlp(self, N, seed=0):
        lbs = []
        for i in range(N):
            print(self.DM.description, 'DLP sample path: ', i)
            t0 = time()
            lb = 0
            dlp = deterministic_lp(self.DM, self.L)
            dlp.DM.rg = RandomState(i+seed)
            while(len(dlp.DM.Horizon)>0):
                print('Horizon: ', len(dlp.DM.Horizon), '  ', end='\r')
                if time() - t0 > TIME_LIMIT:
                    lb = -1
                    break
                lb += dlp.multi_dlp_solve()
                dlp.DM.generate_arrival_departure()
            lbs.append(lb)
            self.write_row(i, 'dlp_lb', lb)


    def sim_ow_lp(self, N, ow_instance, seed=0):
        lbs = []
        olp = ow_lp(ow_instance)
        olp.get_x()
        self.results['olp_ub'] = [olp.upper_bound]
        for i in range(N):
            t0 = time()
            lb = 0
            olp_i = copy.deepcopy(olp)
            olp_i.dm_instance.rg = RandomState(i+seed)
            while(len(olp_i.dm_instance.Horizon)>0):
                if time() - t0 > TIME_LIMIT:
                    lb = -1
                    break
                lb += olp_i.randomized_lp_decision()
                olp_i.dm_instance.generate_arrival_departure()
            lbs.append(lb)
        lb_ = sum(lbs)/N


    def run(self, N):
        self.get_alp_ub()
        self.get_dlp_ub()
        os.system('cls')
        self.sim_alp_dual(N)
        self.sim_alp_primal(N)
        self.sim_alp_primal_no_re(N)
        self.sim_limited_lookahead(N)
        self.sim_dlp(N)

