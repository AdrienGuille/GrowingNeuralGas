import numpy as np
from scipy import spatial
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler

__authors__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class GrowingNeuralGas:

    def __init__(self, input_data):
        self.units = []
        self.connections = []
        self.error = []
        self.data = input_data

    def find_nearest_units(self, observation):
        distance = []
        for unit in self.units:
            distance.append(spatial.distance.euclidean(unit, observation))
        ranking = np.argsort(distance).tolist()
        return ranking

    def update_connections(self, s_1, s_2):
        # update edges emanating from s_1
        edges = self.connections[s_1]
        updated_edges = []
        connected = False
        for unit, age in edges:
            if unit == s_2:
                updated_edges.append((unit, 0))
                connected = True
            else:
                updated_edges.append((unit, age))
        if not connected:
            updated_edges.append((s_2, 0))
        self.connections[s_1] = updated_edges
        # update edges emanating from s_2
        edges = self.connections[s_2]
        updated_edges = []
        for unit, age in edges:
            if unit == s_1:
                updated_edges.append((unit, 0))
            else:
                updated_edges.append((unit, age))
        if not connected:
            updated_edges.append((s_1, 0))
        self.connections[s_2] = updated_edges

    def prune_connections(self, a_max):
        obsolete_units = []
        for i in range(len(self.connections)):
            edges = self.connections[i]
            updated_edges = []
            for unit, age in edges:
                if age < a_max:
                    updated_edges.append((unit, age))
            self.connections[i] = updated_edges
            if len(updated_edges) == 0:
                obsolete_units.append(i)
        for i in sorted(obsolete_units, reverse=True):
            del self.units[i]
            del self.connections[i]
            del self.error[i]

    def train(self, e_b, e_n, a_max, l, a, d, iterations=10):
        # start with two units a and b at random position w_a and w_b
        w_a = [np.random.uniform(-2, 2) for _ in range(2)]
        w_b = [np.random.uniform(-2, 2) for _ in range(2)]
        self.units.append(w_a)
        self.units.append(w_b)
        print(self.units)
        self.error.append(0)
        self.error.append(0)
        self.connections.append([])
        self.connections.append([])
        # iterate through the data
        sequence = 0
        for iteration in range(iterations):
            steps = 0
            np.random.shuffle(data)
            for observation in self.data:
                # find the nearest unit s_1 and the second nearest unit s_2
                nearest_units = self.find_nearest_units(observation)
                s_1 = nearest_units[0]
                s_2 = nearest_units[1]
                # increment the age of all edges emanating from s_1
                edges = self.connections[s_1]
                updated_edges = []
                for unit, age in edges:
                    updated_edges.append((unit, age + 1))
                self.connections[s_1] = updated_edges
                # add the squared distance between the observation and the nearest unit in input space
                self.error[s_1] += spatial.distance.euclidean(observation, self.units[s_1])**2
                # move s_1 and its direct topological neighbors towards the observation by the fractions
                # e_b and e_n, respectively, of the total distance
                update_w_s_1 = e_b * (np.subtract(observation, self.units[s_1]))
                self.units[s_1] = np.add(self.units[s_1], update_w_s_1)
                update_w_s_n = e_n * (np.subtract(observation, self.units[s_1]))
                for neighbor, age in self.connections[s_1]:
                    self.units[neighbor] = np.add(self.units[neighbor], update_w_s_n)
                # if s_1 and s_2 are connected by an edge, set the age of this edge to zero
                # if such an edge doesn't exist, create it
                self.update_connections(s_1, s_2)
                # remove edges with an age larger than a_max
                # if this results in units having no emanating edges; remove them as well
                self.prune_connections(a_max)
                # if the number of steps so far is an integer multiple of parameter l, insert a new unit
                steps += 1
                if steps % l == 0:
                    self.plot_network('visualization/' + str(sequence) + '.png')
                    sequence += 1
                    # determine the unit q with the maximum accumulated error
                    q = np.argmax(self.error)
                    # insert a new unit r halfway between q and its neighbor f with the largest error variable
                    f = -1
                    largest_error = -1
                    for unit, age in self.connections[q]:
                        if self.error[unit] > largest_error:
                            largest_error = self.error[unit]
                            f = unit
                    w_r = 0.5 * (np.add(self.units[q], self.units[f]))
                    r = len(self.units)
                    # insert edges connecting the new unit r with q and f
                    # remove the original edge between q and f
                    self.units.append(w_r)
                    self.connections.append([(q, 0), (f, 0)])
                    edges_q = self.connections[q]
                    updated_edges_q = []
                    for unit, age in edges_q:
                        if unit != f:
                            updated_edges_q.append((unit, age))
                    updated_edges_q.append((r, 0))
                    self.connections[q] = updated_edges_q
                    edges_f = self.connections[f]
                    updated_edges_f = []
                    for unit, age in edges_f:
                        if unit != f:
                            updated_edges_f.append((unit, age))
                    updated_edges_f.append((r, 0))
                    self.connections[f] = updated_edges_f
                    # decrease the error variables of q and f by multiplying them with a
                    self.error[q] *= a
                    self.error[f] *= a
                    # initialize the error variable of r with the new value of the error variable of q
                    self.error.append(self.error[q])
                for i in range(len(self.error)):
                    self.error[i] *= d
        print(self.units)
        print(self.connections)
        print(self.error)

    def plot_network(self, file_path):
        plt.clf()
        plt.scatter(self.data[:, 0], data[:, 1])
        graph = nx.Graph()
        node_pos = {}
        for j in range(len(self.units)):
            node_pos[j] = (self.units[j][0], self.units[j][1])
        for idx in range(len(self.connections)):
            for unit, age in gng.connections[idx]:
                graph.add_edge(idx, unit)
        nx.draw(graph, pos=node_pos)
        plt.draw()
        # plt.show()
        plt.savefig(file_path)

if __name__ == '__main__':
    n_samples = 4000
    dataset_type = 'circles'
    data = None
    if dataset_type == 'moons':
        data = datasets.make_blobs(n_samples=n_samples, random_state=8)
    elif dataset_type == 'blobs':
        data = datasets.make_moons(n_samples=n_samples, noise=.05)
    elif dataset_type == 'circles':
        data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    data = StandardScaler().fit_transform(data[0])
    gng = GrowingNeuralGas(data)
    gng.train(e_b=0.3, e_n=0.006, a_max=10, l=25, a=0.5, d=0.995, iterations=1)

