from math import log
from numpy import mean

class Loss:
    def __init__(self):
        pass

    def loss(y, probability):
        return mean(y * log(probability) + (1 - y) * log(1 - probability))

