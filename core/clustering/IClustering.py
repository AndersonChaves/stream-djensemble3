
class IClustering():
    def __init__(self, config):
        raise NotImplementedError
        
    def run(self):
        raise NotImplementedError
    
    @property
    def clustering_matrix(self):        
        return NotImplementedError
    
    @property
    def embedding_matrix(self):        
        return NotImplementedError
