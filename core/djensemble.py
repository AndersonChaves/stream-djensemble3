from data.dataset_manager import DatasetManager
from query.continuous_query import ContinuousQuery
import logging
import core.utils as ut
from models.models_manager import ModelsManager

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DJEnsemble:
    # --------------------------------------------------------------
    # Main High Level Functions ------------------------------------
    # --------------------------------------------------------------
    def __init__(self, config):        
        logger.info("Initializing Models")
        self.config = config
        self.ds = DatasetManager(self.config["data_source"])        
        self.ds.loadDataset((self.config["time_start"], self.config["time_end"]))
        self.models_manager = ModelsManager(config["models"])
        self.t = 0                

    def run(self):
        self.run_offline()
        self.run_online()

    # --------------------------------------------------------------
    # Main Steps ---------------------------------------------------
    # --------------------------------------------------------------
    def run_offline(self):                
        self.update_cost_estimation_function()
        self.initialize_continuous_query()

    def update_cost_estimation_function(self):
        logger.debug("Updating Cost Estimation Function")        
        self.load_clustering_data()
        self.models_manager.update_cef()

    def initialize_continuous_query(self):
        logger.debug("Initializing Query")
        self.continuous_query = ContinuousQuery(self.config)
        self.continuous_query.prepare(
            self.pre_clustering_window, self.models_manager.get_models())

    def load_clustering_data(self):
        self.pre_clustering_ds = DatasetManager(self.config["data_source"])
        self.pre_clustering_ds.loadDataset(
            (0, self.config["clustering"]["pre_clustering_window_size"]))
        self.pre_clustering_window = self.pre_clustering_ds.read_all_data()

    def run_online(self):        
        while(self.read_window()):
            self.run_query()               

    def read_window(self):
        logger.info(f"Reading Window {self.t} to {self.t+self.config['window_size']}")
        self.data_window = self.ds.read_window(self.t, self.config["window_size"]+1)
        self.t += 1
        if len(self.data_window) == self.config["window_size"]+1:
            return True
        elif len(self.data_window) <= self.config["window_size"]+1:
            return False
        else:
            raise Exception("")
        

    def run_query(self):        
        logger.info("Running Query")
        self.continuous_query.run(
            input=self.data_window[:-1], true_output=self.data_window[-1])
        logger.info(self.continuous_query.get_statistics())        

    def update_error(self):
        self.continuous_query.update_error(self.data_window[-1])

    def get_parameters(self):
        self.text = f"Parameters:"
        self.text += \
            f"   Embedding: {self.config['clustering']['embedding_method']}\n"
        self.text += \
            f"   Clustering Type: {self.config['clustering']['type']}\n"
        self.text += \
            f"   Cluster Query Window: {self.config['clustering']['cluster_query_window']}\n" 
        self.text += \
            f"   Tiling: {self.config['tiling']['strategy']}\n"
        return self.text
        


