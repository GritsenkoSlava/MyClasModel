from numpy import mean

class Counting:
    def __init__(self):
        pass

    def gradient(self, probability, x, y):
        grad_w = mean((probability - y) * x)
        grad_b = mean(probability - y)

        return grad_w, grad_b 