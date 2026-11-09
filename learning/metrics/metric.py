from math import log, e

class Metrics:
    def __init__(self):
        pass

    def BCE(self, y, p):
        return y * log(p) + (1 - y) * log(1 - p)