import csv
import math
from random import random

# help function to read inputs in csv files
def read_csv(filename):
    res = []
    with open(filename, newline='') as csvfile:
        file = csv.reader(csvfile)
        for row in file:
            res.append(row)
    return res

# help function to transpose a matrix represented by list of lists
def transpose_mat(mat):
    res = list()
    row, col = len(mat), len(mat[0])
    for m in range(0, col):
        temp = list()
        for n in range(0, row):
            temp.append(mat[n][m])
        res.append(temp)
    return res

# help function to add matrices represented by list of lists
def add_mat(mat1, mat2):
    mat1_row, mat1_col = len(mat1), len(mat1[0])
    mat2_row, mat2_col = len(mat2), len(mat2[0])
    # check the dimensions of input matrices
    if (mat1_row != mat2_row) or (mat1_col != mat2_col):
        print("Incorrect dimentions of input matrices")
        return None
    # calculate the add
    res = [[0 for col in range(0, mat1_col)] for row in range(0, mat1_row)]
    for i in range(0, mat1_row):
        for j in range(0, mat1_col):
            res[i][j] += mat1[i][j] + mat2[i][j]
    return res

# help function to multiply to matrices represented by list of lists
def mul_mat(mat1, mat2):
    mat1_row, mat1_col = len(mat1), len(mat1[0])
    mat2_row, mat2_col = len(mat2), len(mat2[0])
    # check the dimensions of input matrices
    if mat1_col != mat2_row:
        print("Incorrect dimentions of input matrices")
        return None
    # calculate the multiplication 
    res = [[0 for row in range(0, mat2_col)] for col in range(0, mat1_row)]
    for i in range(0, mat1_row):
        for j in range(0, mat2_col):
            for k in range(0, mat1_col):
                res[i][j] += mat1[i][k] * mat2[k][j]
    return res

# initialize a neural network layer
def init_layer(n_inputs, n_outputs):
    w = [[random() for i in range(n_outputs)] for i in range(n_inputs)]
    b = [[random() for i in range(n_outputs)]]
    return {'weights': w, 'bias': b}

# initialize a neural network
def init_nn(n_inputs, n_outputs, *n_hidden):
    network = list()
    if len(n_hidden) == 0:
        network.append(init_layer(n_inputs, n_outputs))
    elif len(n_hidden) == 1:
        network.append(init_layer(n_inputs, n_hidden[0]))
        network.append(init_layer(n_hidden[0], n_outputs))
    else:
        network.append(init_layer(n_inputs, n_hidden[0]))
        for n in range(0, len(n_hidden)-1):
            network.append(init_layer(n_hidden[n], n_hidden[n+1]))
        network.append(init_layer(n_hidden[-1], n_outputs))
    return network

# define the linear activation function
def linear_activate(inputs, weights, bias):
    res = add_mat(mul_mat(inputs, weights), bias)
    return res

# define the ReLu transfer function
def relu_transfer(x):
    res = [max(0,i) for i in x]
    return res

# define the softmax transfer function
def softmax_transfer(x):
    x_exp = [math.exp(i) for i in x]
    res = [i/sum(x_exp) for i in x_exp]
    return res

# forward feeding function 
def forward_feeding(inputs, network, transfer):
    res = inputs
    for layer in network:
        res = linear_activate(res, layer['weights'], layer['bias'])
        if transfer=='relu':
            res[0] = relu_transfer(res[0])
        elif transfer=='softmax':
            res[0] = softmax_transfer(res[0])
    return res
    
nn1 = init_nn(3, 2, 1, 10, 15, 2)
nn1o = forward_feeding([[1,2,3]], nn1, 'softmax')
print(nn1o)

t1 = init_nn(2,2)
tr = forward_feeding([[1,2]], t1, 'softmax')

    