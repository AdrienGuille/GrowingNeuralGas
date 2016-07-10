# GrowingNeuralGas

## About

A simple implementation of the Growing Neural Gas algorithm, based on:


    A Growing Neural Gas Network Learns Topologies, B. Fritzke
    Advances in Neural Information Processing Systems 7 (1995)`

## Usage

Instantiate a GNG object with some data, then fit the neural network:

    gng = GrowingNeuralGas(data)
    gng.fit_network(e_b=0.1, e_n=0.006, a_max=10, l=200, a=0.5, d=0.995, passes=8, plot_evolution=True)

## Example

This example shows (i) how to generate toy data (two interleaving half circles) using the scikit-learn lib and (ii) how to perform cluster analysis through the growing neural gas network:

    from gng import GrowingNeuralGas
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    
    print('Generating data...')
    data = datasets.make_moons(n_samples=2000, noise=.05) 
    data = StandardScaler().fit_transform(data[0])
    print('Done.')
    print('Fitting neural network...')
    gng = GrowingNeuralGas(data)
    gng.fit_network(e_b=0.1, e_n=0.006, a_max=10, l=200, a=0.5, d=0.995, passes=8, plot_evolution=True)
    print('Found %d clusters.' % gng.number_of_clusters())
    gng.plot_clusters(gng.cluster_data())


Running example.py will produce the following output:

    Preparing data...
    Done.
    Fitting neural network...
       Pass #1
       Pass #2
       Pass #3
       Pass #4
       Pass #5
       Pass #6
       Pass #7
       Pass #8
    Found 2 clusters.


### Clusters
![alt tag](https://raw.githubusercontent.com/AdrienGuille/GrowingNeuralGas/visualization/clusters.png)

### Network properties
![alt tag](https://raw.githubusercontent.com/AdrienGuille/GrowingNeuralGas/visualization/network_properties.png)

### Global error vs. number of passes
![alt tag](https://raw.githubusercontent.com/AdrienGuille/GrowingNeuralGas/visualization/global_error.png)

### Accumulated local error vs. total number of iterations
![alt tag](https://raw.githubusercontent.com/AdrienGuille/GrowingNeuralGas/visualization/accumulated_local_error.png)

