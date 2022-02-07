import networkx as nx
from dm_data import dm_instance
import numpy as np
import math
from simulator import simulator
import pandas as pd

'''
    ridesharing instance generator
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
            if e[0][1] == e[1][1]:
                self.f[e] = 0
            elif eval(e[0][1]) + eval(e[1][1]) == 3:
                self.f[e] = 0.01
            elif eval(e[0][1]) + eval(e[1][1]) == 4:
                self.f[e] = 1
            elif eval(e[0][1]) + eval(e[1][1]) == 5:
                self.f[e] = 0.02
        self.w = {}
        for i in self.G.nodes:
            self.w[i] = 0

        a_dm = dm_instance(self.G, self.t, self.alpha, self.beta, self.initState, self.f, self.r, self.w)
        return a_dm



class ridesharing_instance:
    def __init__(self, N, t):
        '''
        N: # of driver types indexed by n
        J: # of customer types indexed by j
        r(n,j): reward for match j to n
        f(n,j): failure probability of match (n,j)
        arr_d(n): arrival rate of driver type n [Length N list]
        arr_c(j): arrival rate of custoemr type j [Length J list]
        '''
        self.N = N
        self.t = t
        self.density = .4 #density of traffic network

    def generate_instance(self, seed=42):
        N = self.N
        i = seed
        while True:
            self.Network = nx.gnp_random_graph(N, self.density, seed=i)
            i += 100
            if nx.is_connected(self.Network):
                break
        self.G = nx.DiGraph()
        for n in range(N):
            '''
                each region has two driver classes: High and Low
                                two customer classes: High and Low
            '''
            self.G.add_node('DH'+str(n+1))
            self.G.add_node('DL'+str(n+1))
            self.G.add_node('CH'+str(n+1))
            self.G.add_node('DL'+str(n+1))

        self.r, self.f = {}, {}

        for i in range(N):
            for j in range(N):
                self.G.add_edge('DH'+str(j+1), 'CH'+str(i+1))
                self.G.add_edge('CH'+str(i+1), 'DH'+str(j+1))

                self.G.add_edge('DL'+str(j+1), 'CL'+str(i+1))
                self.G.add_edge('CL'+str(i+1), 'DL'+str(j+1))

                self.G.add_edge('DH'+str(i+1), 'CH'+str(j+1))
                self.G.add_edge('CH'+str(j+1), 'DH'+str(i+1))

        for e in self.G.edges:
            self.r[e] = 0
            self.f[e] = 0

        np.random.seed(seed)
        for i in range(N):
            for j in range(N):
                self.r[('DH'+str(j+1), 'CH'+str(i+1))] = np.random.uniform(0.25,0.30)
                self.r[('DL'+str(j+1), 'CH'+str(i+1))] = np.random.uniform(0.8,0.12)
                self.r[('DL'+str(j+1), 'CL'+str(i+1))] = np.random.uniform(0.8,0.12)

                if i == j:
                    self.f[('DH'+str(j+1), 'CH'+str(i+1))] = 0
                    self.f[('DL'+str(j+1), 'CL'+str(i+1))] = 0
                    self.f[('DL'+str(j+1), 'CH'+str(i+1))] = 0.1
                else:
                    L = len(nx.shortest_path(self.Network,source=i,target=j))
                    if L == 1:
                        self.f[('DH'+str(j+1), 'CH'+str(i+1))] = np.random.uniform(low=0.01, high=0.05)
                        self.f[('DL'+str(j+1), 'CL'+str(i+1))] = np.random.uniform(low=0.01, high=0.05)
                        self.f[('DL'+str(j+1), 'CH'+str(i+1))] = np.random.uniform(low=0.01, high=0.05)
                    elif L == 2:
                        self.f[('DH'+str(j+1), 'CH'+str(i+1))] = np.random.uniform(low=0.4, high=0.6)
                        self.f[('DL'+str(j+1), 'CL'+str(i+1))] = np.random.uniform(low=0.4, high=0.6)
                        self.f[('DL'+str(j+1), 'CH'+str(i+1))] = np.random.uniform(low=0.6, high=0.8)
                    else:
                        self.f[('DH'+str(j+1), 'CH'+str(i+1))] = np.random.uniform(low=0.95, high=0.99)
                        self.f[('DL'+str(j+1), 'CL'+str(i+1))] = np.random.uniform(low=0.95, high=0.99)
                        self.f[('DL'+str(j+1), 'CH'+str(i+1))] = np.random.uniform(low=0.95, high=0.99)

        self.alpha, self.beta = {}, {}
        DH_arr, DL_arr, CH_arr, CL_arr = {}, {}, {}, {}
        for i in range(N):
            DH_arr[i] = np.random.rand()
            DL_arr[i] = np.random.rand()*3
            CH_arr[i] = np.random.rand()
            CL_arr[i] = np.random.rand()*3
        D_arr_sum = sum(DH_arr.values()) + sum(DL_arr.values())
        C_arr_sum = sum(CH_arr.values()) + sum(CL_arr.values())

        arrivals_per_period = 20
        D_sojourn, C_sojourn = 15, 5
        for i in range(N):
            self.beta['DH'+str(i+1)] = arrivals_per_period*DH_arr[i]/D_arr_sum
            self.beta['DL'+str(i+1)] = arrivals_per_period*DL_arr[i]/D_arr_sum
            self.beta['CH'+str(i+1)] = arrivals_per_period*CH_arr[i]/C_arr_sum
            self.beta['CL'+str(i+1)] = arrivals_per_period*CL_arr[i]/C_arr_sum

            self.alpha['DH'+str(i+1)] = 1 - np.exp(-D_sojourn)
            self.alpha['DL'+str(i+1)] = 1 - np.exp(-D_sojourn)
            self.alpha['CH'+str(i+1)] = 1 - np.exp(-C_sojourn)
            self.alpha['CL'+str(i+1)] = 1 - np.exp(-C_sojourn)

        self.initState = {}
        w = {}
        for v in self.G.nodes:
            w[v] = 0
            self.initState[v] = np.random.randint(10)
        a_dm = dm_instance(self.G, self.t, self.alpha, self.beta, self.initState, self.f, self.r, w)
        return a_dm



if __name__ == '__main__':
    ride_pdf = pd.DataFrame()
    ride_generator = ow_instance(1)
    ride_dm = ride_generator.generate_instance()
    sim = simulator(ride_dm, 2)
    sim.run(100)
    sim.results['description'] = ['ow_1']
    sim.get_result_pdf()
    print(sim.result_pdf)
    ride_pdf = ride_pdf.append(sim.result_pdf, ignore_index=True)


    ride_generator = ow_instance(10)
    ride_dm = ride_generator.generate_instance()
    sim = simulator(ride_dm, 2)
    sim.run(10)
    sim.results['description'] = ['ow_10']
    sim.get_result_pdf()
    print(sim.result_pdf)

    ride_pdf = ride_pdf.append(sim.result_pdf, ignore_index=True)

    ride_generator = ow_instance(100)
    ride_dm = ride_generator.generate_instance()
    sim = simulator(ride_dm, 2)
    sim.run(10)
    sim.results['description'] = ['ow_100']
    sim.get_result_pdf()
    print(sim.result_pdf)

    ride_pdf = ride_pdf.append(sim.result_pdf, ignore_index=True)

    for i in range(10):
        ride_generator = ridesharing_instance(5, 20)
        ride_dm = ride_generator.generate_instance(i)
        sim = simulator(ride_dm, 2)
        sim.run(10)
        sim.results['description'] = ['ride_5_20_'+str(i+1)]
        sim.get_result_pdf()
        print(sim.result_pdf)

        ride_pdf = ride_pdf.append(sim.result_pdf, ignore_index=True)


    for i in range(10):
        ride_generator = ridesharing_instance(10, 100)
        ride_dm = ride_generator.generate_instance(i)
        sim = simulator(ride_dm, 2)
        sim.run(10)
        sim.results['description'] = ['ride_10_100_'+str(i+1)]
        sim.get_result_pdf()
        print(sim.result_pdf)

        ride_pdf = ride_pdf.append(sim.result_pdf, ignore_index=True)

    for i in range(10):
        ride_generator = ridesharing_instance(20, 100)
        ride_dm = ride_generator.generate_instance(i)
        sim = simulator(ride_dm, 2)
        sim.run(10)
        sim.results['description'] = ['ride_20_100_'+str(i+1)]
        sim.get_result_pdf()
        print(sim.result_pdf)

        ride_pdf = ride_pdf.append(sim.result_pdf, ignore_index=True)

    ride_pdf.to_csv('results/ride.csv')  







