# coding: utf-8

import numpy as np
from scipy import spatial
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import os
import shutil

__authors__ = 'Adrien Guille'
__email__ = 'adrien.guille@univ-lyon2.fr'

'''
Simple implementation of the Growing Neural Gas algorithm, based on:
A Growing Neural Gas Network Learns Topologies. B. Fritzke, Advances in Neural
Information Processing Systems 7, 1995.
'''


class GrowingNeuralGas:

    def __init__(self, input_data):
        self.units = []
        self.network = None
        self.error = []
        self.data = input_data
        plt.style.use('ggplot')

    def find_nearest_units(self, observation):
        distance = []
        for unit in self.units:
            distance.append(spatial.distance.euclidean(unit, observation))
        ranking = np.argsort(distance).tolist()
        return ranking

    def prune_connections(self, a_max):
        for u, v, attributes in self.network.edges(data=True):
            if attributes['age'] > a_max:
                self.network.remove_edge(u, v)
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                self.network.remove_node(u)

    def fit_network(self, e_b, e_n, a_max, l, a, d, iterations=100):
        # logging variables
        accumulated_global_error = []
        network_order = []
        # 0. start with two units a and b at random position w_a and w_b
        w_a = [np.random.uniform(-2, 2) for _ in range(2)]
        w_b = [np.random.uniform(-2, 2) for _ in range(2)]
        self.units.append(w_a)
        self.units.append(w_b)
        self.error.append(0)
        self.error.append(0)
        self.network = nx.Graph()
        self.network.add_node(0)
        self.network.add_node(1)
        # 1. iterate through the data
        sequence = 0
        steps = 0
        np.random.shuffle(data)
        for observation in self.data:
            if steps == iterations:
                break
            # 2. find the nearest unit s_1 and the second nearest unit s_2
            nearest_units = self.find_nearest_units(observation)
            s_1 = nearest_units[0]
            s_2 = nearest_units[1]
            # 3. increment the age of all edges emanating from s_1
            for u, v, attributes in self.network.edges_iter(data=True, nbunch=[s_1]):
                self.network.add_edge(u, v, age=attributes['age']+1)
            # 4. add the squared distance between the observation and the nearest unit in input space
            self.error[s_1] += spatial.distance.euclidean(observation, self.units[s_1])**2
            # 5 .move s_1 and its direct topological neighbors towards the observation by the fractions
            #    e_b and e_n, respectively, of the total distance
            update_w_s_1 = e_b * (np.subtract(observation, self.units[s_1]))
            self.units[s_1] = np.add(self.units[s_1], update_w_s_1)
            update_w_s_n = e_n * (np.subtract(observation, self.units[s_1]))
            for neighbor in self.network.neighbors(s_1):
                self.units[neighbor] = np.add(self.units[neighbor], update_w_s_n)
            # 6. if s_1 and s_2 are connected by an edge, set the age of this edge to zero
            #    if such an edge doesn't exist, create it
            self.network.add_edge(s_1, s_2, age=0)
            # 7. remove edges with an age larger than a_max
            #    if this results in units having no emanating edges, remove them as well
            self.prune_connections(a_max)
            # 8. if the number of steps so far is an integer multiple of parameter l, insert a new unit
            steps += 1
            if steps % l == 0:
                self.plot_network('visualization/sequence/' + str(sequence) + '.png')
                sequence += 1
                # 8.a determine the unit q with the maximum accumulated error
                q = np.argmax(self.error)
                # 8.b insert a new unit r halfway between q and its neighbor f with the largest error variable
                f = -1
                largest_error = -1
                for u in self.network.neighbors(q):
                    if self.error[u] > largest_error:
                        largest_error = self.error[u]
                        f = u
                w_r = 0.5 * (np.add(self.units[q], self.units[f]))
                r = len(self.units)
                # 8.c insert edges connecting the new unit r with q and f
                #     remove the original edge between q and f
                self.units.append(w_r)
                self.network.add_edge(r, q, age=0)
                self.network.add_edge(r, f, age=0)
                self.network.remove_edge(q, f)
                # 8.d decrease the error variables of q and f by multiplying them with a
                #     initialize the error variable of r with the new value of the error variable of q
                self.error[q] *= a
                self.error[f] *= a
                self.error.append(self.error[q])
            # 9. decrease all error variables by multiplying them with a constant d
            accumulated_global_error.append(np.sum(self.error))
            network_order.append(self.network.size())
            for i in range(len(self.error)):
                self.error[i] *= d
        plt.clf()
        plt.xlabel("iterations")
        plt.plot(range(len(accumulated_global_error)), accumulated_global_error, label='global error')
        plt.plot(range(len(network_order)), network_order, label='number of units')
        plt.legend()
        plt.savefig('visualization/global_error_and_network_size.png')

    def plot_network(self, file_path):
        plt.clf()
        plt.scatter(self.data[:, 0], self.data[:, 1])
        node_pos = {}
        for j in range(len(self.units)):
            node_pos[j] = (self.units[j][0], self.units[j][1])
        nx.draw(self.network, pos=node_pos)
        plt.draw()
        plt.savefig(file_path)


if __name__ == '__main__':
    if os.path.exists('visualization/sequence'):
        shutil.rmtree('visualization/sequence')
    os.makedirs('visualization/sequence')
    n_samples = 2000
    dataset_type = 'moons'
    data = None
    print('Preparing data...')
    if dataset_type == 'blobs':
        data = datasets.make_blobs(n_samples=n_samples, random_state=8)
    elif dataset_type == 'moons':
        data = datasets.make_moons(n_samples=n_samples, noise=.05)
    elif dataset_type == 'circles':
        data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    data = StandardScaler().fit_transform(data[0])
    print('Done.')
    print('Fitting neural network...')
    gng = GrowingNeuralGas(data)
    gng.fit_network(e_b=0.3, e_n=0.006, a_max=5, l=20, a=0.5, d=0.995, iterations=1000)
    print('Found %d clusters:' % nx.number_connected_components(gng.network))
