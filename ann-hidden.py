from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
feature_set, labels = datasets.make_moons(noise=0.1)
plt.figure(figsize=(10,7))
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)
#plt.show()

labels = labels.reshape(100, 1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

np.random.seed(42)
wh = np.random.rand(len(feature_set[0]), 4) #shape(2, 4)
wo = np.random.rand(4, 1) #shape(4, 1)
lr = 0.5
#bias = np.random.rand(1)


for epoch in range(20000):
    # feedforward step
    zh = np.dot(feature_set, wh) #shape (100, 4)
    ah = sigmoid(zh) #shape (100, 4)

    zo = np.dot(ah, wo) #shape (100, 1)
    ao = sigmoid(zo) #shape (100, 1)
    
    # backpropagation step 1
    error_out = (1 / 2) * (np.power((ao - labels), 2))
    print(error_out.sum())

    dcost_dao = ao - labels #shape (100, 1)
    dao_dzo = sigmoid_der(zo) #shape (100, 1)
    dzo_dwo = ah #shape (100, 4)

    dcost_dwo = np.dot(dzo_dwo.T, dao_dzo * dcost_dao) #shape ((4, 100), (100, 1) * (100, 1)) = (4, 1)

    # backpropagation step 2
    dzo_dah = wo #shape (4, 1)
    dcost_dah = np.dot(dao_dzo * dcost_dao,  dzo_dah.T) #shape((100, 1) * (100, 1), (1, 4)) = (100, 4)
    dah_dzh = sigmoid_der(zh) #shape(100, 4)
    dzh_dwh = feature_set #shape(100, 2)
    dcost_dwh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah) #shape ((2, 100), (100, 4) * (100, 4)) = (2, 4)

    # update weights
    wh -= lr * dcost_dwh
    wo -= lr * dcost_dwo
