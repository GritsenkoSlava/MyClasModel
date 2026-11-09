from abc import ABC, abstractmethod

class Optimizer(ABC):

    @abstractmethod
    def get_grad(self):
        pass

    @abstractmethod
    def step(self):
        pass