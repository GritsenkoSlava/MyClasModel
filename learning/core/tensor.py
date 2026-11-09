import pandas as pd
import numpy as np
class Tensor:
    def __init__(self):
        self.data = pd.read_csv("/home/koturo/train_data/train_xy_clas.csv")

    def processing_data(self):
        data_in_dt = pd.DataFrame(self.data)

        x_data = data_in_dt.iloc[:, 0].to_numpy()
        y_data = data_in_dt.iloc[:, 1].to_numpy()

        x_data = x_data.reshape(-1, 2)
        y_data = y_data.reshape(-1, 2)

        return x_data, y_data