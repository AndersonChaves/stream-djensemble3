import numpy as np

class IEmbeddingStrategy():
    def iterate(self, data: np.ndarray):
        raise NotImplementedError
    
    def get_description(self):
        raise NotImplementedError