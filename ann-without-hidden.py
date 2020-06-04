import numpy as np
feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
labels = np.array([[1],[0],[0],[1],[1]])


np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sig_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


for epochs in range(20000):
    inputs = feature_set
    XW = np.dot(inputs, weights) + bias
    z = sigmoid(XW)
    error = z - labels
    #e = (z - labels) * (z - labels)
    print(error.sum())
    dcost_dpred = error
    dpred_dz = sig_der(z)
    z_delta = dcost_dpred * dpred_dz

    inputs = inputs.T
    weights = weights - lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias = bias - lr* num


def test(x):
    x = np.array(x)
    XW = np.dot(x, weights) + bias
    result = sigmoid(XW)
    print(result)



