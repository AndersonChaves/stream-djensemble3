import numpy as np

def test():
    print("OK")

class NoiseGenerator():
    def add_noise(self, dataset: np.array):
        for cell in np.nditer(dataset, op_flags=['readwrite']):
            cell += 1
        return dataset
