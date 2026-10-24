import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from learning.core.tensor import Tensor
from learning.engine.grad import Grad
from learning.engine.loss import Loss
from learning.engine.update import Update
from numpy import array, float32
from math import e


class Model:
    def __init__(self):
        self.tensor = Tensor()
        self.grad = Grad()
        self.loss = Loss()
        self.update = Update()

        self.x, self.y = self.tensor.process_data()
        self.w, self.b = 0.0, 0.0

    def learn(self, epochs, learning_rate):
        for epoch in range(epochs):
            logit = self.x * self.w + self.b
            probability = 1 / (1 + e ** (-logit))
            loss = self.loss.loss(self.y, probability)
            w_grad, b_grad = self.grad.gradient(probability, self.x, self.y)
            self.w, self.b = self.update.update(self.w, self.b, w_grad, b_grad, learning_rate)

            if (epoch + 1) % 1000 == 0:
                print(f'Loss {loss:.6f}')

    def predict(self, digit):
        test_digit = array([[digit]], dtype=float32)

        logit = self.w * test_digit + self.b
        probability = 1 / 1 (1 + e ** -logit)

        predict = (probability >= 0.5)
        predict = 1 if predict == True else 0

        print(f'Probability: {probability}  ||'
              f'  Predict {predict}')

model = Model()

model.learn(10000, 0.01)
print('\n')
model.predict(4)