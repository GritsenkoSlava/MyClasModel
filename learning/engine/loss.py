from math import log
from numpy import mean

class Loss:
    def __init__(self):
        pass

    def loss(self, y, probability):
        return mean(y * probability + (1 - y) * (1 - probability))
