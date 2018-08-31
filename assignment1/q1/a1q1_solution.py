# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 17:51:56 2018

@author: P1312812
"""
import csv

# Define a help function to read in csv files
def read_csv(filename):
    res = []
    with open(filename, newline='') as csvfile:
        file = csv.reader(csvfile)
        for row in file:
            res.append(row)
    return res

# Read in weights and bias to initialize a neural network
def init_nn(weights_file, bias_file, n_nodes, n_layers):
    # read in csv files and filter out heading columns
    weights = list(map(lambda x: x[1:], read_csv(weights_file)))
    bias = list(map(lambda x: x[1:], read_csv(bias_file)))
    # initialize a neural network
    network = list()
#    print(n_layers == len(bias))
#    print(n_nodes == int(len(weights)/len(bias)))
    for i in range(0, n_layers):
        print("--- layer {}->{} --- ".format(str(i), str(i+1)))
        layer = [{'weights': weights[i*n_nodes:i*n_nodes+5],
                  'bias': bias[i]}]
        print(layer)
        network.append(layer)
#    print(n_nodes)
#    print(n_layers)
#    print(weights)
#    print(len(weights))
#    print(bias)
#    print(len(bias))
    return network

nn = init_nn("./a/a_w.csv", "./a/a_b.csv", 5, 3)

def combine_two_layers(layer1, layer2):
    res = 0
    print("layer1")
    print(layer1)
    print("layer2")
    print(layer2)
    return res

tt = combine_two_layers(nn[0], nn[1])

#from random import seed
#from random import random
## Initialize a network
#def initialize_network(n_inputs, n_hidden, n_outputs):
#    network = list()
#    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
#    network.append(hidden_layer)
#    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
#    network.append(output_layer)
#    return network
# 
#seed(1)
#network = initialize_network(5, 1, 5)
#for layer in network:
#    print(layer)
    
    