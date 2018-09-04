import csv
import math
from random import seed
from random import random

seed(1)

# help function for lists multiplication
def mul_list(list1, list2):
    if (len(list1) != len(list2)):
        print("Input lists have different lengths")
        return None
    res = list()
    for i in range(0, len(list1)):
        res.append(list1[i]*list2[i])
    return res
#print(mul_list([1,2,3], [1,0,1]))
#-> [1, 0, 3]

# help function for matrix transpose
def transpose_mat(mat):
    res = list()
    row, col = len(mat), len(mat[0])
    for n in range(0, col):
        temp = list()
        for m in range(0, row):
            temp.append(mat[m][n])
        res.append(temp)
    return res

# help function for matrix scalar multiplication
def sca_mat(mat, scalar):
    res = list()
    row, col = len(mat), len(mat[0])
    for m in range(0, row):
        temp = list()
        for n in range(0, col):
            temp.append(mat[m][n]*scalar)
        res.append(temp)
    return res
#print(sca_mat([[1,2,3], [1,1,4]],2))
#-> [[2, 4, 6], [2, 2, 8]]

# help function for matrices addition
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
#print(add_mat([[1,2,3]], [[0,1,0]]))
#-> [[1, 3, 3]]
#print(add_mat([1,3,4],[0,2,1]))
#-> error

# help function for matrices substraction
def sub_mat(mat1, mat2):
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
            res[i][j] += mat1[i][j] - mat2[i][j]
    return res
#print(sub_mat([[1,2,3]], [[0,1,0]]))
#-> [[1, 1, 3]]

# help function for matrices multiplication
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
#t_inputs = [[1,2]]
#t_weights = [[0.5,1,2], [1,0.5,0.5]]
#print(mul_mat(t_inputs, t_weights))
#-> [[2.5, 2.0, 3.0]]
    

# help function to create the identity matrix
def identity_mat(n):
    res = list()
    for i in range(n):
        temp = []
        for j in range(n):
            if i==j:
                temp.append(1)
            else:
                temp.append(0)
        res.append(temp)
    return res
#print(identity_mat(2))
#-> [[1, 0], [0, 1]]

# initialize a neural network layer
def init_layer(n_inputs, n_outputs):
    w = [[random()/10 for i in range(n_outputs)] for i in range(n_inputs)]
    b = [[random()/10 for i in range(n_outputs)]]
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
#t_inputs = [[1,2]]
#t_weights = [[0.5,1,2], [1,0.5,0.5]]
#t_bias = [[0.1,0.1,0.2]]
#print(linear_activate(t_inputs, t_weights, t_bias))
#-> [[2.6, 2.1, 3.2]]

# define the ReLu transfer function
def relu_transfer(x):
    res = [max(0,i) for i in x]
    return res

# calculate the derivative of Relu
def relu_transfer_derivative(x):
    res = list()
    for i in x:
        if i>0:
            res.append(1)
        else:
            res.append(0)
    return res
#print(relu_transfer_derivative([-1,2,3,-8]))
#-> [0, 1, 1, 0]

# define the softmax transfer function
def softmax_transfer(x):
    x_exp = [round(math.exp(i),16) for i in x]
    res = [i/sum(x_exp) for i in x_exp]
    return res
#print(softmax_transfer([-1,1,2,3]))
#-> [0.012037642711939451, 0.08894681729740428, 0.24178251715880078, 0.6572330228318555]

# calculate the derivative of softmax
def softmax_transfer_derivative(x):
    x_exp = [round(math.exp(i),16) for i in x]
    res = [i*(sum(x_exp)-i)/(sum(x_exp)**2) for i in x_exp]
    return res
#print(softmax_transfer_derivative([1,2,3,4]))
#-> [0.03103084923581511, 0.0795501864530196, 0.18076934858369267, 0.22928868580089717]

# forward feeding function 
def forward_feeding(inputs, network):
    res = inputs.copy()
    for n in range(len(network)):
        layer = network[n]
        res = linear_activate(res, layer['weights'], layer['bias'])
        if n==(len(network)-1):
            res[0] = softmax_transfer(res[0])
        else:
            res[0] = relu_transfer(res[0])
        layer['output'] = res
    return res

### testting
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# test backpropagation of error
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)
#for layer in network:
#	print(layer)
#-> [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': -0.0005348048046610517}]
#-> [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': -0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': 0.0771723774346327}]

def sigmoid_transfer_derivative(x):
    res = [i * (1.0 - i) for i in x]
    return res
#print(sigmoid_transfer_derivative([0.1,0.2,0.3]))
#-> [0.09000000000000001, 0.16000000000000003, 0.21]

# backward propagate function
def backward_propagate(network, expected):
    for n in reversed(range(len(network))):
        layer = network[n]
        errors = list()
        if n==(len(network)-1):
            errors = sub_mat(expected, layer['output'])
            derivative = softmax_transfer_derivative(layer['output'][0])
        else:
            layer_plus = network[n+1]
            errors = mul_mat(layer_plus['delta'],
                             transpose_mat(layer_plus['weights']))
            derivative = relu_transfer_derivative(layer['output'][0])
        delta = mul_list(errors[0], derivative)
        layer['delta'] = [delta]
    pass

network = [{'weights':[[0.13436424411240122], [0.8474337369372327]], 'bias':[[0.763774618976614]], 'output':[[0.7105668883115941]]},
            {'weights':[[0.2550690257394217, 0.4494910647887381]], 'bias':[[0.49543508709194095, 0.651592972722763]], 'output':[[0.6213859615555266, 0.6573693455986976]]}]
expected = [[0, 1]]
backward_propagate(network, expected)
#for layer in network:
#    print(layer)
#-> {'weights': [[0.13436424411240122], [0.8474337369372327]], 'bias': [[0.763774618976614]], 'output': [[0.7105668883115941]], 'delta': [[-0.0005348048046610517]]}
#-> {'weights': [[0.2550690257394217, 0.4494910647887381]], 'bias': [[0.49543508709194095, 0.651592972722763]], 'output': [[0.6213859615555266, 0.6573693455986976]], 'delta': [[-0.14619064683582808, 0.0771723774346327]]}

# update the neural network weights
def update_weights(network, inputs, l_rate):
    for n in range(len(network)):
        if n!=0:
            inputs = network[n-1]['output']
        temp = sca_mat(mul_mat(transpose_mat(inputs),
                               network[n]['delta']), l_rate)
        new_weights = add_mat(temp,network[n]['weights'])
        temp = sca_mat(network[n]['delta'], l_rate)
        new_bias = add_mat(temp,network[n]['bias'])
        network[n]['weights'] = new_weights
        network[n]['bias'] = new_bias
    return None

# calculate cross entropy cost
def cross_entropy_cost(pred_res, actual_res):
    cost = 0
    n = len(pred_res)
    for i in range(n):
        cost += (actual_res[i]*math.log(pred_res[i])
                 + (1-actual_res[i])*math.log(1-pred_res[i]))
    cost = (-1/n)*cost
    return cost
#print(cross_entropy_cost([0.3541008934987469, 0.1548374930445365, 0.18574050171086504, 0.30532111174585147],
#                         [0,0,1,0]))
#-> 0.6632621566107613

def square_error(pred_res, actual_res):
    cost = 0
    n = len(pred_res)
    for i in range(n):
        cost += (actual_res[i]-pred_res[i])**2
    return cost
#print(square_error([1,2,3],[1,1,2]))
#-> 2

# train the neural network
def train_nn(network, train_inputs, expected_outputs, l_rate, n_epoch):
    for epoch in range(n_epoch):
        sum_error = 0
        for n in range(len(train_inputs)):
            outputs = forward_feeding(train_inputs[n], network)
            real = expected_outputs[n]
            error = cross_entropy_cost(outputs[0], real[0])
            sum_error += error
            backward_propagate(network, real)
            update_weights(network, train_inputs[n], l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# help function to read inputs in csv files
def read_x_csv(filename):
    res = []
    with open(filename, newline='') as csvfile:
        file = csv.reader(csvfile)
        for row in file:
            rec = list()
            for e in row:
                rec.append(int(e))
            res.append([rec])
    return res

# one hot encoding
def one_hot_encoding(inputs, n_class):
    res = list()
    for i in inputs:
        temp = list()
        for k in range(0, n_class):
            if i==k:
                temp.append(1)
            else:
                temp.append(0)
        res.append([temp])
    return res
#print(one_hot_encoding([1,2,3,0,1],4))
#-> [[[0, 1, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 0, 1]], [[1, 0, 0, 0]], [[0, 1, 0, 0]]]
    
def read_y_csv(filename):
    res = list()
    with open(filename, newline='') as csvfile:
        file = csv.reader(csvfile)
        for row in file:
            for e in row:
                res.append(int(e))
    return res

# Part 1
x_train = read_x_csv('x_train.csv')
y_train = one_hot_encoding(read_y_csv('y_train.csv'), 4)
x_test = read_x_csv('x_test.csv')
y_test = one_hot_encoding(read_y_csv('y_test.csv'), 4)
network_a = init_nn(14, 4, 100, 40)
#train_nn(network_a, x_train[0:10], y_train[0:10], 0.01, 10)
#train_nn(network_a, x_train[0:1000], y_train[0:1000], 0.001, 10)

network_b = init_nn(14, 4, 28, 28, 28, 28, 28, 28, 40)
#train_nn(network_b, x_train[0:10], y_train[0:10], 0.01, 10)
#train_nn(network_b, x_train, y_train, 0.01, 10)

network_c = init_nn(14, 4, 14, 14, 14, 14, 14, 14, 14,
                    14, 14, 14, 14, 14, 14, 14,
                    14, 14, 14, 14, 14, 14, 14,
                    14, 14, 14, 14, 14, 14, 14)
#train_nn(network_c, x_train[0:10], y_train[0:10], 0.01, 10)

# Part 2
def read_wb_csv(filename, ):
    res = {}
    with open(filename, newline='') as csvfile:
        file = csv.reader(csvfile)
        for row in file:
            key = row[0]
            if key not in res.keys():
                res[key] = []
            res[key].append([float(e) for e in row[1:]])
    return res

#X=[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]
#label = [3]
#weights = read_wb_csv('./b/w-100-40-4.csv')
#bias = read_wb_csv('./b/b-100-40-4.csv')
#nn = list()
#for i in range(len(weights.keys())):
#    layer = {}
#    layer['weights'] = weights[list(weights.keys())[i]]
#    layer['bias'] = bias[list(bias.keys())[i]]
#    nn.append(layer)
#ipt = [X]
#opt = forward_feeding(ipt, nn)
#actual = one_hot_encoding(label, 4)[0]
#error = cross_entropy_cost(opt[0], actual[0])

#train_nn(nn, [X], one_hot_encoding(label, 4), 0.001, 10)

#dataset = [[2.7810836,2.550537003,0],
#	[1.465489372,2.362125076,0],
#	[3.396561688,4.400293529,0],
#	[1.38807019,1.850220317,0],
#	[3.06407232,3.005305973,0],
#	[7.627531214,2.759262235,1],
#	[5.332441248,2.088626775,1],
#	[6.922596716,1.77106367,1],
#	[8.675418651,-0.242068655,1],
#	[7.673756466,3.508563011,1]]
#n_inputs = len(dataset[0]) - 1
#n_outputs = len(set([row[-1] for row in dataset]))
#network = init_nn(n_inputs, 2, n_outputs)
#train_inputs = list()
#for rec in dataset:
#    train_inputs.append([rec[0:2]])
#expected_opts = list()
#for rec in dataset:
#    expected_opts.append([[0,1]])
#train_nn(network, train_inputs, expected_opts, 0.001, 50)
#for layer in network:
#	print(layer)
    