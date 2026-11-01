from pandas import *
class Tensor:
    def __init__(self):
        self.data = read_csv("/home/koturo/train_data/train_xy_clas.csv")

    def __enter__(self):
        return f'Data: {self.data}'

    def process_data(self):
        data = DataFrame(self.data)

        x = data.iloc[:, 0].to_numpy(dtype=float)
        y = data.iloc[:, 1].to_numpy(dtype=float)

        return x, y 

        