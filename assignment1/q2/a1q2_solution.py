import csv
import numpy as np

def init_layer(n_inputs, n_outputs):
    w = np.random.randn(n_inputs, n_outputs)/np.sqrt(n_outputs)
    b = np.random.randn(n_outputs)
    return {'weights': w, 'bias': b}
#print(init_layer(2,3))

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
nn = init_nn(2,3,4,3)

# define the linear activation function
def linear_activate(inputs, weights, bias):
    res = np.add(np.dot(inputs, weights), bias)
    return res
#t_inputs = [[1,2]]
#t_weights = [[0.5,1,2], [1,0.5,0.5]]
#t_bias = [[0.1,0.1,0.2]]
#print(linear_activate(t_inputs, t_weights, t_bias))
#-> [[2.6, 2.1, 3.2]]

# define the ReLu transfer function
def relu_transfer(x):
    return np.maximum(x, 0)

## calculate the derivative of Relu
def relu_transfer_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
#print(relu_transfer_derivative([-1,2,3,-8]))
#-> [0, 1, 1, 0]

# define the softmax transfer function
def softmax_transfer(x):
    e_x = np.exp(x - np.max(x))
    e_x_sum = e_x.sum(axis=0)
    return e_x / e_x_sum
#print(softmax_transfer(np.array([-1,1,2,3])))
#-> [0.012037642711939451, 0.08894681729740428, 0.24178251715880078, 0.6572330228318555]

# calculate the derivative of softmax
def softmax_transfer_derivative(x):
    e_x = np.exp(x - np.max(x))
    e_x_sum = e_x.sum(axis=0)
    return e_x*(e_x_sum-e_x)/(e_x_sum**2)
#print(softmax_transfer_derivative([1,2,3,4]))
#-> [0.03103084923581511, 0.0795501864530196, 0.18076934858369267, 0.22928868580089717]

# forward feeding function 
def forward_feeding(inputs, network):
    res = inputs
    for n in range(len(network)):
        layer = network[n]
        res = linear_activate(res, layer['weights'], layer['bias'])
        if n==(len(network)-1):
            res[0] = softmax_transfer(res[0])
        else:
            res[0] = relu_transfer(res[0])
        layer['output'] = res
    return res

# backward propagate function
def backward_propagate(network, expected):
    for n in reversed(range(len(network))):
        layer = network[n]
        #errors = list()
        if n==(len(network)-1):
            errors = np.subtract(expected, layer['output'])
            derivative = softmax_transfer_derivative(layer['output'][0])
        else:
            layer_plus = network[n+1]
            errors = np.dot(layer_plus['delta'], layer_plus['weights'].T)
            derivative = relu_transfer_derivative(layer['output'][0])
        delta = np.multiply(errors[0], derivative)
        layer['delta'] = [delta]
    pass

# update the neural network weights
def update_weights(network, inputs, l_rate):
    for n in range(len(network)):
        if n!=0:
            inputs = network[n-1]['output']
        temp = np.multiply(np.dot(inputs.T, network[n]['delta']),l_rate)
        new_weights = np.add(temp, network[n]['weights'])
        temp = np.multiply(network[n]['delta'],l_rate)
        new_bias = np.add(temp, network[n]['bias'])
        network[n]['weights'] = new_weights
        network[n]['bias'] = new_bias
    return None

def cross_entropy_cost(pred, actual):
    return -(np.sum(np.multiply(actual, np.log10(pred))) +
             np.sum(np.multiply((1-actual), np.log10(1-pred))))
#p = np.array([0.98, 0.01, 0.01])
#a = np.array([1.0, 0.0, 0.0])
#print(np.multiply(a, np.log10(p)))
#print(np.multiply((1-a), np.log10(1-p)))
#print(cross_entropy_cost(p,a))
#-> 0.6632621566107613

# train the neural network
def train_nn(network, x_train, y_train, x_test, y_test, l_rate, n_epoch):
    train_n_inputs = len(x_train)
    test_n_inputs = len(x_test)
    
#    shuffle_train_index = np.arange(train_n_inputs)
#    np.random.shuffle(shuffle_train_index)
#    x_train = x_train[shuffle_train_index]
#    y_train = y_train[shuffle_train_index]
    
    res = {'stat':[], 'trained_nn': None}
    for epoch in range(n_epoch):
        train_error_sum = 0
        train_correct = 0
        test_error_sum = 0
        test_correct = 0
        for m in range(train_n_inputs):
            current_input = x_train[m]
            pred_output = forward_feeding(current_input, network)
            actual_output = y_train[m]
            train_error_sum += cross_entropy_cost(pred_output[0],
                                                  actual_output[0])
            if np.argmax(pred_output[0]) == np.argmax(actual_output[0]):
                train_correct += 1
            backward_propagate(network, actual_output)
            update_weights(network, current_input, l_rate)
        for n in range(test_n_inputs):
            current_input = x_test[n]
            pred_output = forward_feeding(current_input, network)
            actual_output = y_test[n]
            test_error_sum += cross_entropy_cost(pred_output[0],
                                                 actual_output[0])
            if np.argmax(pred_output[0]) == np.argmax(actual_output[0]):
                test_correct += 1
        rec = ('>epoch=%d, lrate=%.3f, train_error=%.3f, train_acc=%.3f, ' +
               'test_error=%.3f, test_acc=%.3f') % (epoch+1, l_rate,
                train_error_sum/train_n_inputs, train_correct/train_n_inputs,
                test_error_sum/test_n_inputs, test_correct/test_n_inputs)
        print(rec)
        res['stat'].append([rec])
    res['trained_nn'] = network
    return res

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

def write_csv(filename, res):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for w in res:
            writer.writerow(w)
    
# Part 1
#np.random.seed(11)
x_train = np.array(read_x_csv('x_train.csv'))
y_train = np.array(one_hot_encoding(read_y_csv('y_train.csv'), 4))
x_test = np.array(read_x_csv('x_test.csv'))
y_test = np.array(one_hot_encoding(read_y_csv('y_test.csv'), 4))
network_a = init_nn(14, 4, 100, 40)
#train_nn(network_a, x_train[0:10], y_train[0:10], 0.001, 10)
#k = 20
#train_nn(network_a, x_train[0:k], y_train[0:k], x_test[0:k], y_test[0:k], 0.001, 200)
#res_a = train_nn(network_a, x_train, y_train, x_test, y_test, 0.001,600)
#write_csv('network_a_output_stat.csv', res_a['stat'])

network_b = init_nn(14, 4, 28, 28, 28, 28, 28, 28, 40)
#train_nn(network_b, x_train[0:10], y_train[0:10], 0.01, 10)
#res_b = train_nn(network_b, x_train, y_train, x_test, y_test, 0.001, 200)
#write_csv('network_b_output_stat_600.csv', res_b['stat'])

np.random.seed(13)
network_c = init_nn(14, 4, 14, 14, 14, 14, 14, 14, 14,
                    14, 14, 14, 14, 14, 14, 14,
                    14, 14, 14, 14, 14, 14, 14,
                    14, 14, 14, 14, 14, 14, 14)
#k = 100
#train_nn(network_c, x_train[0:k], y_train[0:k], x_test[0:k], y_test[0:k], 0.1, 500)
res_c = train_nn(network_c, x_train, y_train, x_test, y_test, 0.001, 200)
write_csv('network_c_output_stat_200.csv', res_c['stat'])

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


    