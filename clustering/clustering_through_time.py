from data.dataset_manager import DatasetManager
from clustering.static_clustering import StaticClustering
from clustering.stream_clustering import StreamClustering
import core.view as view
import time
import logging
from statistics import mean


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ClusteringThroughTime():
    def __init__(self, config):
        self.config = config
        self.experiment_complete = False
        self.stream_clustering = None
        self.embedding_method = self.config["clustering"]["embedding_method"]

    def run(self):
        self.time_start = time.time()
        start, end = self.config["time_start"], self.config["time_end"]
        step = self.config["window_size"]
        self.silhouette_history = []
        for t in range(start, end, step):
            logger.debug(f"Performing clustering: ds from {t} to {t+step} \n")
            if self.config["type"] == "static":
                silhouette = self.perform_static_clustering(t, step)
                logger.debug(f"Silhouette: {silhouette}")
                self.silhouette_history.append(silhouette)
            else:
                silhouette = self.perform_stream_clustering(t, step)
                logger.debug(f"Silhouette: {silhouette}")
                self.silhouette_history.append(silhouette)
        self.time_end = time.time()
        self.experiment_complete = True


    def load_dataset(self):
        self.ds = DatasetManager(self.config["clustering"]["data_source"])
        self.ds.loadDataset()
        
    def perform_static_clustering(self, t, step):
        self.load_dataset()
        data = self.ds.read_window(t, step)
        clustering = StaticClustering(self.config["clustering"])
        clustering.run()
        return clustering.silhouette
    
    def perform_stream_clustering(self, t, step):
        self.load_dataset()
        data = self.ds.read_window(t, step)
        if self.stream_clustering is None:
            self.stream_clustering = StreamClustering(self.embedding_method)
            self.stream_clustering.initialize_clustering(data)
        else:
            self.stream_clustering.update_clustering(data)
        return self.stream_clustering.get_silhouette()

    def get_parameters(self):
        stats = f"Clustering Through Time: {self.config['type']}\n"
        stats += f"Embedding: {self.embedding_method} \n"
        stats += f"Time Range: {self.config['time_start']}, {self.config['time_end']}\n"
        stats += f"Window Size: {self.config['window_size']}\n"
        return stats

    def get_statistics(self):
        stats = f"Clustering Through Time: {self.config['type']}\n"
        stats += f"Embedding: {self.embedding_method} \n"
        stats += f"total time: {self.time_end - self.time_start} \n"
        stats += f"silhouette history: {self.silhouette_history} \n"
        stats += f"Average Silhouette: {mean(self.silhouette_history)} \n"
        return stats

    def get_silhouette_history(self):
        if self.experiment_complete:
            return self.silhouette_history
        else:
            raise Exception("Error - Experiment not run yet.")