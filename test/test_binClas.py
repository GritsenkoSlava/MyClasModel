import numpy as np 
from math import log, e

def learn(epochs, lr, digit_test):
    x = np.array([[1], [3], [4], [6], [5], [8]])
    y = np.array([[0], [0], [0], [1], [1], [1]])
    w = 0.0
    b = 0.0

    for i in range(epochs):
        logit = x * w + b
        p = 1 / (1 + e ** (-logit))
        #loss = np.mean(y * log(p) + (1 - y) * log(1 - p))

        w_grad = np.mean((p - y) * x)
        b_grad = np.mean(p - y)

        w = w - lr * w_grad
        b = b - lr * b_grad

    digit = np.array([[digit_test]], dtype=np.float32)
    logit = w * x + b
    probality = 1 / (1 + e ** (-logit))
    predict = (probality >= 0.5)

    print(f'Probality: {probality}', '\n',
          f'Class: {predict}')

learn(10000, 0.001, 4)

