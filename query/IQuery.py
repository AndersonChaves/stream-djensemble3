import numpy as np

class IQuery():
    def __init__(self, config):
        return NotImplementedError

    def prepare(self, pre_clustering_window: np.ndarray, 
                candidate_models):
        return NotImplementedError
    
    def run(self, input: np.ndarray):        
        return NotImplementedError
        
    def update_error(self, true_output: np.ndarray):
        return NotImplementedError
    
    def get_statistics(self):
        return NotImplementedError