import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from learning.optimizer.optimizer import Optimizer


class SGD(Optimizer):
    
    def get_grad(self, x, y, p):
        w_grad = (p - y) * x
        b_grad = (p - y)

        return w_grad, b_grad

    def step(self, old_w, old_b, learning_rate, w_grad, b_grad):
        w_new = old_w - learning_rate * w_grad
        b_new = old_b - learning_rate * b_grad

        return w_new, b_new