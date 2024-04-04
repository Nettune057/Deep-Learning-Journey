import sys
import os
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import training as tn

import function.activation_function as act_func
data = pd.read_csv('../datasets/mnist_train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = tn.init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = tn.forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = tn.backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = tn.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2



def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = tn.forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
if __name__ == "__main__":
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

    test_prediction(0, W1, b1, W2, b2)
    test_prediction(1, W1, b1, W2, b2)
    test_prediction(2, W1, b1, W2, b2)
    test_prediction(3, W1, b1, W2, b2)

    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    get_accuracy(dev_predictions, Y_dev)