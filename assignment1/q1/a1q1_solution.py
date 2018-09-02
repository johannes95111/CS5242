import csv

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

# read in weights and bias to initialize a neural network
def init_nn(weights_file, bias_file, n_nodes, n_layers):
    # read in csv files and filter out heading columns
    weights = list(map(lambda x: x[1:], read_csv(weights_file)))
    bias = list(map(lambda x: x[1:], read_csv(bias_file)))
    # convert str to float
    for i in range(0, len(weights)):
        weights[i] = list(map(lambda x: float(x), weights[i]))
    for i in range(0, len(bias)):
        bias[i] = list(map(lambda x: float(x), bias[i]))
    # initialize a neural network
    network = list()
    for i in range(0, n_layers):
        w = weights[i*n_nodes:(i+1)*n_nodes]
        b = list()
        b.append(bias[i])
        layer = {'weights': w, 'bias': b}
        network.append(layer)
    return network

# help function to multiply two lists
def list_mul(list1, list2):
    res = list()
    for n2 in list2:
        temp = 0
        for n1 in list1:
            temp += n1*n2
        res.append(temp)
    return res        

# combine two layers into one layer
def combine_two_layers(layer1, layer2):
    # get the weights and bias
    weights1, bias1 = layer1['weights'], layer1['bias']
    weights2, bias2 = layer2['weights'], layer2['bias']
    # combine the weights
    res_weights = mul_mat(weights1, weights2)
    # combine the bias
    res_bias = add_mat(mul_mat(bias1, weights2), bias2)    
    return {"weights": res_weights, "bias": res_bias}

# help function to write outputs in csv files
def write_csv(filename, res):
    with open(filename+'-w.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for w in res['weights']:
            writer.writerow(w)
    with open(filename+'-b.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for b in res['bias']:
            writer.writerow(b)

# forward feeding function
def forward_feeding(inputs, network):
    res = inputs
    for layer in network:
        res = add_mat(mul_mat(res, layer['weights']), layer['bias'])
    return res


# read in files and execute the functions to transfer
nn = init_nn("./test/a_w.csv", "./test/a_b.csv", 2, 2)
print(forward_feeding([[1,2]], nn))
t1 = combine_two_layers(nn[0], nn[1])
print(forward_feeding([[1,2]], [t1]))
#t2 = combine_two_layers(t1, nn[2])
write_csv('test', t1)

nn = init_nn("./a/a_w.csv", "./a/a_b.csv", 5, 3)
print(forward_feeding([[1,2,3,4,5]], nn))
temp = combine_two_layers(nn[0], nn[1])
res = combine_two_layers(temp, nn[2])
print(forward_feeding([[1,2,3,4,5]], [res]))
write_csv('./a/a', res)

#nn = init_nn("./b/b_w.csv", "./b/b_b.csv", 5, 3)
#temp = combine_two_layers(nn[0], nn[1])
#res = combine_two_layers(temp, nn[2])
#write_csv('./b/b', res)
#
#nn = init_nn("./c/c_w.csv", "./c/c_b.csv", 5, 3)
#temp = combine_two_layers(nn[0], nn[1])
#res = combine_two_layers(temp, nn[2])
#write_csv('./c/c', res)
#
#nn = init_nn("./d/d_w.csv", "./d/d_b.csv", 5, 3)
#temp = combine_two_layers(nn[0], nn[1])
#res = combine_two_layers(temp, nn[2])
#write_csv('./d/d', res)
#
#nn = init_nn("./e/e_w.csv", "./e/e_b.csv", 5, 3)
#temp = combine_two_layers(nn[0], nn[1])
#res = combine_two_layers(temp, nn[2])
#write_csv('./e/e', res)
    