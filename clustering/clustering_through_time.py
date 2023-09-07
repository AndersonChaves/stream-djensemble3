from data.dataset_manager import DatasetManager
from clustering.static_clustering import StaticClustering
from clustering.stream_clustering import StreamClustering
from clustering.embedding import get_embedded_series_representation
from clustering.parcorr import get_parcorr_series_representation_from_dataset
import core.view as view
import numpy as np
import time
import logging
from statistics import mean
from sklearn.metrics import silhouette_score
from random import random
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ClusteringThroughTime():
    def __init__(self, config):
        self.config = config
        self.experiment_complete = False
        self.static_clustering = None
        self.stream_clustering = None
        self.embedding_method = self.config["clustering"]["embedding_method"]
        self.window_size = self.config["window_size"]
        self.n_clusters = self.config["n_clusters"]

    def run(self):
        self.time_start = time.time()
        start, end = self.config["time_start"], self.config["time_end"]
        if self.config["window_type"] == 'tumbling':
            step = self.window_size
        elif self.config["window_type"] == 'sliding':
            step = 1
        self.silhouette_history = []        
        for t in range(start, end, step):
            if t+step > end:
                break
            logger.debug(f"Performing clustering: ds from {t} to {t+self.window_size} \n")
            cluster_start = datetime.now()
            if self.config["type"] == "dynamic":
                silhouette = self.perform_dynamic_clustering(t, self.window_size)
            elif self.config["type"] == "static":
                silhouette = self.perform_static_clustering(t, self.window_size)
            elif self.config["type"] == "stream":
                silhouette = self.perform_stream_clustering(t, self.window_size)
            clustering_time = datetime.now() - cluster_start
            logger.info(f"Silhouette: {silhouette} t={t}")
            logger.info(f"Clustering Time: {clustering_time}")
            logger.info(f"Estimated time: {(end - t) / step * clustering_time}")
            self.silhouette_history.append(silhouette)
        self.time_end = time.time()
        self.experiment_complete = True
        self.average_silhouette = mean(self.silhouette_history)
        self.average_silhouette_2nd_half = mean(self.silhouette_history[int(len(self.silhouette_history)/2):])

    def load_dataset(self, start, end):
        self.ds = DatasetManager(self.config["clustering"]["data_source"])
        #self.ds.set_range()
        self.ds.loadDataset((start, end))

    def perform_static_clustering(self, t, step):
        self.load_dataset(t, t+step)
        data = self.ds.read_all_data()
        if self.static_clustering is None:
            self.static_clustering = StaticClustering(data, 
                self.embedding_method, n_clusters=self.config["n_clusters"])
            self.static_clustering.run()
        else:
            self.static_clustering.predict(data)     
        if t % 10 == 0:               
            self.static_clustering.save_clustering_image(f"images/static/{self.embedding_method}/w={self.window_size}", f"{t}-{t+step}")
        return self.static_clustering.silhouette

    def perform_dynamic_clustering(self, t, step):
        self.load_dataset(t, t+step)
        data = self.ds.read_all_data()
        self.dynamic_clustering = StaticClustering(data, self.embedding_method, 
            n_clusters=self.config["n_clusters"])
        self.dynamic_clustering.run()
        if t % 10 == 0:               
            self.dynamic_clustering.save_clustering_image(f"images/dynamic/{self.embedding_method}/w={self.window_size}", f"{t}-{t+step}")
        return self.dynamic_clustering.silhouette
    
    def perform_stream_clustering(self, t, step):
        self.load_dataset(t, t+step)
        data = self.ds.read_all_data()
        if self.stream_clustering is None:
            self.stream_clustering = StreamClustering(self.embedding_method, 
                n_clusters=self.config["n_clusters"])
            self.stream_clustering.initialize_clustering(data)
        else:
            self.stream_clustering.update_clustering(data)
        n_clusters = self.stream_clustering.number_of_clusters
        if t % 10 == 0:               
            self.stream_clustering.save_clustering_image(f"images/stream/{self.embedding_method}/w={self.window_size}", f"{t}-{t+step}")
        return self.stream_clustering.get_silhouette()

    def get_parameters(self):
        stats = f"Clustering Through Time: {self.config['type']}\n"
        stats += f"Embedding: {self.embedding_method} \n"
        stats += f"Time Range: {self.config['time_start']}, {self.config['time_end']}\n"
        stats += f"Window Size: {self.config['window_size']}\n"
        return stats

    def get_statistics(self):
        stats = f"total time: {self.time_end - self.time_start} \n"
        stats += f"silhouette history: {self.silhouette_history} \n"
        stats += f"Average Silhouette: {mean(self.silhouette_history)} \n"
        stats += f"Average second half: \
            {mean(self.silhouette_history[int(len(self.silhouette_history)/2):])} \n"
        return stats

    def get_silhouette_history(self):
        if self.experiment_complete:
            return self.silhouette_history
        else:
            raise Exception("Error - Experiment not run yet.")