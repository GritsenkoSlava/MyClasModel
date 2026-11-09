import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from learning.core.tensor import Tensor
from learning.metrics.metric import Metrics
from learning.optimizer.sgd import SGD
from numpy import array
from math import e

class Model:
    def __init__(self):
        self.Tensor    = Tensor()
        self.Loss      = Metrics()
        self.Optimizer = SGD()

        self.x, self.y = self.Tensor.processing_data()
        self.w, self.b = array([[0.0]]), array([[0.0]])

    def learn(self, epochs, learning_rate):
        for epoch in range(epochs+1):
            logit = self.w * self.x + self.b
            probability = 1 / 1 + e ** -logit

            grad_w, grad_b = self.Optimizer.get_grad(self.x, logit, probability)
            #loss = self.Loss.BCE(logit, probability)
            self.w, self.b = self.Optimizer.step(self.w, self.b, learning_rate, grad_w, grad_b)

            #if (epoch + 1) % 100 == 0:
            #    print(f"||  Loss:  {loss:.8f}  ||")

    def predict(self, testDigitX):
        digit = array([[testDigitX]], dtype=float)
        logit = self.w * digit + self.b
        probability = 1 / (1 + e ** (-logit))
        predict = (probability >= 0.5)
        predict = 1 if predict is True else 0

        print(f'||  Probability  {probability}  ||'
              f'  Class (1/0)  {predict}  ||')
