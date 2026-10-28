import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from learning.core.tensor import Tensor
from learning.engine.optimizer import Optimizer
from learning.engine.loss import Loss
from learning.engine.count_grad import Counting
from numpy import array, float32
from math import e

class Model:
    def __init__(self):
        self.tensor    = Tensor()
        self.optimizer = Optimizer()
        self.count     = Counting()
        self.loss      = Loss()

        self.w, self.b = array([[0]], dtype=float32), array([[0]], dtype=float32)
        self.x, self.y = self.tensor.process_data()

    def learn(self, epochs, learning_rate):
        for epoch in range(epochs):
            logit = self.w @ self.w + self.b
            probability = 1 / (1 + e ** -logit)
            loss = self.loss.loss(self.y, probability)

            w_grad, b_grad = self.count.gradient(probability, self.x, self.y)
            self.w, self.b = self.optimizer.step(self.w, self.b, w_grad, b_grad, learning_rate)

            if (epoch + 1) % 100 == 0:
                print(f'Loss:  {loss}')

        logit = self.w * self.x + self.b
        probability = 1 / (1 + e ** -logit)
        predict = (probability >= 0.5)
        predict = 1 if predict is True else 0

        '''
        if predict is True:
            predict = 1
        else:
            predict = 0
        '''

        print(f'  Probability: {probability}  ')

    '''def test(self, test_digit):
        digit = array([[4]], dtype=float32)
        logit = self.w @ digit + self.b
        probability = 1 / (1 + e ** - logit)
        predict = (probability >= 0.5)
        predict = 1 if predict is True else 0

        print(probability)
        print(predict)'''

model = Model()
model.learn(2000, 0.001)
