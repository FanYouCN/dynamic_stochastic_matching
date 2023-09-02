import networkx as nx
from dm_data import dm_instance
import numpy as np
import math
from simulator import simulator
import pandas as pd
from affine_model import affine_model
import argparse
from temp import parse_df




'''
    matchmaking instance generator
'''

def generate_mm(T, seed=42, wait_coef=0.5, arr_coef=1.0):
    # base and first 3 extensions
    np.random.seed(seed)
    g = nx.Graph()
    for l in ['W','N','M','S']:
        for s in ['h','m','l']:
            g.add_node(l+s)

    for i in g.nodes:
        for j in g.nodes:
            if not (i[0] + j[0]) in ["WN", "NW"] :
                if not (i[1] + j[1] ) in ["hl", "lh"] :
                    g.add_edge(i,j)   
    r = {}
    reward_coef = 1
    for e in g.edges:
        diff = (e[0][0]==e[1][0]) + 2*(e[1][0]==e[1][1])
        if diff == 0:
            r[e] = 10 * np.random.normal(loc=1.0, scale=0.1) * reward_coef  
        elif diff == 1:
            r[e] = 8 * np.random.normal(loc=1.0, scale=0.1) * reward_coef  
        elif diff == 2:
            r[e] = 6 * np.random.normal(loc=1.0, scale=0.1) * reward_coef  
        else:
            r[e] = 5 * np.random.normal(loc=1.0, scale=0.1) * reward_coef  
    w = {}
    for i in g.nodes:
        if i[1] == "h":
            w[i] = 2 * np.random.normal(loc=1.0, scale=0.1) * wait_coef
        elif i[1] == "m":
            w[i] = 3 * np.random.normal(loc=1.0, scale=0.1) * wait_coef
        else:
            w[i] = 4 * np.random.normal(loc=1.0, scale=0.1) * wait_coef
    s0 = {} 
    nu_bar = {}
    for i in g.nodes:
        s0[i] = np.random.randint(2)
        nu_bar[i] = 1
    arr = {}
    for i in g.nodes:
        if i[1] == 'h':
            arr[i] = 0.2 * np.random.normal(loc=1.0, scale=0.2) * arr_coef
        elif i[1] == 'm':
            arr[i] = 0.5 * np.random.normal(loc=1.0, scale=0.2) * arr_coef
        else:
            arr[i] = 0.3 * np.random.normal(loc=1.0, scale=0.2) * arr_coef
    mu = {}
    for e in g.edges:
        mu[e] = np.random.uniform(0, 0.001)
    a_dm = dm_instance(g, T, nu_bar, arr, s0, mu, r, w, seed)
    return a_dm


def generate_mm_instance(num_regions, num_skill_levels, len_horizon, seed=42):
    # larger instances
    rg = np.random.RandomState(seed=seed)
    connection_quality = {}
    for i in range(num_regions):
        for j in range(num_regions):
            if i != j:
                quality = rg.choice(['good', 'ok', 'bad'], p=[0.5,0.3,0.2])
                connection_quality[i,j] = quality
                connection_quality[j,i] = quality
            else:
                connection_quality[i,j] = 'good'

    player_type_labels = {}
    G = nx.DiGraph()
    v = 0
    for i in range(num_regions):
        for k in range(num_skill_levels):
            player_type = [i] + [k]
            G.add_node(v)
            player_type_labels[v] = player_type
            v += 1
    reward = {}
    for i in G:
        for j in G:
            connection = connection_quality[player_type_labels[i][0], player_type_labels[j][0]]
            skill_diff = abs(player_type_labels[i][1] - player_type_labels[j][1])
            if skill_diff*2 <= num_skill_levels:
                G.add_edge(i,j)
                G.add_edge(j,i)
    for e in G.edges:
        i, j = e[0], e[1]
        connection = connection_quality[player_type_labels[i][0], player_type_labels[j][0]]
        skill_diff = abs(player_type_labels[i][1] - player_type_labels[j][1])
        if connection == 'good':
            reward[i,j] = max(0.1, rg.normal(loc=(1-skill_diff/num_skill_levels)*1, scale=0.1))
        if connection == 'ok':
            reward[i,j] = max(0.1, rg.normal(loc=(1-skill_diff/num_skill_levels)*.7, scale=0.1))
        if connection == 'bad':
            reward[i,j] = max(0.1, rg.normal(loc=(1-skill_diff/num_skill_levels)*.3, scale=0.1))
    nu_bar, lam = {}, {}
    arr = {}
    for i in G:
        nu_bar[i] = 1 - np.exp(-10) 
        arr[i] = rg.randint(10)
    arr_sum = sum(arr.values())
    for i in G:
        lam[i] = arr[i]/arr_sum*100
    mu = {}
    for e in G.edges:
        mu[e] = rg.uniform(0, 0.02)
    init_state = {}
    w = {}
    for i in G:
        init_state[i] = rg.randint(5)
        w[i] = rg.uniform(0, 0.1)
    dm = dm_instance(G, len_horizon, nu_bar, lam, init_state, mu, reward, w, seed)
    return dm

