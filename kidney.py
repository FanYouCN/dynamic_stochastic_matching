import networkx as nx
from dm_data import dm_instance
import numpy as np
import math
import random
from helper import are_blood_compatible
from simulator import simulator
import pandas as pd

'''
    kidney exchange instance generator (Saidman generator)
'''

class kidney_instance:
    def __init__(self, arrival, sojourn, horizon):
        G = nx.DiGraph()
        nodes, edges = set(), set()
        r, f = {}, {}
        for patient_gender in ['M', 'F']:
            for is_spouse in ['Y','N']:
                for patient_BT in ['O','A','B','AB']:
                    for donor_BT in ['O','A','B','AB']:
                        for patientPRA_type in ['H', 'L']:
                            node_label = patient_gender+is_spouse+patient_BT+'_'+donor_BT+'-'+patientPRA_type
                            nodes.add(node_label)
        self.all_types = list(nodes)
        for i in nodes: #self loops for already blood type compatible pairs with failure rate
            patient_BT = i[2:i.find('_')]
            donor_BT = i[i.find('_')+1:i.find('-')]
            if are_blood_compatible(patient_BT, donor_BT):
                e = (i, i)
                edges.add(e)
                r[e] = random.gauss(37.1506024096, 22.2170610307)
                patientPRAtype = i[i.find('-')+1:]
                patient_gender = i[0]
                is_spouse = i[1]
                isWifePatient = (patient_gender == 'F') and (is_spouse == 'Y')
                this_falure_rate = 0
                if isWifePatient:
                    if patientPRA_type == 'H':
                        this_falure_rate = 1.0 - 0.75 * (1.0 - 0.90)
                    else:
                        this_falure_rate = 1.0 - 0.75 * (1.0 - 0.05)
                else:
                    if patientPRA_type == 'H':
                        this_falure_rate = 0.90
                    else:
                        this_falure_rate = 0.05
                f[e] = this_falure_rate
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                patient_BT = i[2:i.find('_')]
                donor_BT = j[i.find('_')+1:i.find('-')]
                if are_blood_compatible(patient_BT, donor_BT):
                    e = (i, j)
                    edges.add(e)
                    r[e] = random.gauss(37.1506024096, 22.2170610307)
                    patientPRA_type = i[i.find('-')+1:]
                    this_falure_rate = 0
                    if patientPRA_type == 'H':
                        this_falure_rate = 0.90
                    else:
                        this_falure_rate = 0.05
                    f[e] = this_falure_rate
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        initState = {}
        w = {}
        for i in G.nodes:
            initState[i] = random.randint(0,6)
            w[i] = 0
        alpha, beta = {}, {}
        for i in G.nodes:
            alpha[i] = 1 - np.exp(-sojourn+1)
        for i in G.nodes:
            prob = 1
            if i[0] == 'M':
                prob *= 1- 0.4090
            else:
                prob *= 0.4090
            if i[1] == 'Y':
                prob *= 0.4897
            else:
                prob *= 1 - 0.4897
            if i[2] == 'O':
                prob *= 0.4814
            elif i[2] == 'A':
                prob *= 0.3373
            elif i[2] == 'B':
                prob *= 0.1428
            else:
                prob *= 1 - 0.4814 - 0.3373 - 0.1428
            if i[3] == 'O':
                prob *= 0.4814
            elif i[3] == 'A':
                prob *= 0.3373
            elif i[3] == 'B':
                prob *= 0.1428
            else:
                prob *= 1 - 0.4814 - 0.3373 - 0.1428
            if i[4] == 'H':
                prob *= 0.25
            else:
                prob *= 1-0.25
            beta[i] = prob * arrival
            self.dm = dm_instance(G, horizon, alpha, beta, initState, f, r, w)


if __name__ == '__main__':
    dm = kidney_instance(200,10,24).dm
    sim = simulator(dm, L=2)
    sim.run(10)
    sim.results['description'] = ['kidney_2']
    sim.get_result_pdf()
    print(sim.result_pdf)

    kidney_pdf = sim.result_pdf.copy()



    dm = kidney_instance(200,10,24).dm
    sim = simulator(dm, L=3)
    sim.run(10)
    sim.results['description'] = ['kidney_3']
    sim.get_result_pdf()
    print(sim.result_pdf)

    kidney_pdf = kidney_pdf.append(sim.result_pdf)

    kidney_pdf.to_csv('results/kidney.csv')  