import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from clustering.embedding import *
from data.dataset_manager import DatasetManager
import time
from config.config import Config
import warnings
import sys, os
import core.view as view

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class StaticClustering():
    def __init__(self, config):        
        self.config = config
        self.target_dataset = self._read_target_dataset()
        self.embedding_method = config["embedding_method"]
        self.embedding_strategy = create_embedding_strategy(
            self.target_dataset, self.embedding_method)        
        self._initialize_parameters()
        self.it_number = 1
        self.use_cache = False
        
    def run(self):
        if self.clustering_done:
            raise Exception("Error - Clustering has already been done")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._calculate_embedding()        
            self._cluster()
        self.clustering_done = True                 
    
    def get_statistics(self):
        self._check_clustering()
        stats = f"Clustering: {self.embedding_method} \n"
        stats += f"embedding time: {self.embedding_time} \n"
        stats += f"clustering time: {self.clustering_time} \n"
        stats += f"total time: {self.embedding_time + self.clustering_time} \n"
        stats += f"clustering silhouette: {self.silhouette} \n"
        return stats

    def save_clustering_image(self, file_dir, file_name):
        if file_dir[-1] != "/":
            file_dir += "/"
        view.save_figure_from_matrix(self.clustering_matrix, file_dir, file_name)

    def set_cache(self, it_number:int):
        self.use_cache = True
        self.it_number = it_number

    def _read_target_dataset(self):
        self.ds_manager = DatasetManager(self.config["data_source"])
        self.ds_manager.loadDataset()
        return self.ds_manager.read_all_data()        

    def _initialize_parameters(self):
        self._embedding_matrix = None
        self._clustering_matrix = None
        self._silhouette = -999
        self.clustering_done = False

    def _supress_log_messages(self, switch:str):
        if switch == "on":
            self.original_stdout = sys.stdout
            null_device = open(os.devnull, 'w')
            sys.stdout = null_device
        else:
            sys.stdout = self.original_stdout

    def _calculate_embedding(self):
        start = time.time()
        file_name = f"{self.config['embedding_method']}t{self.config['data_source']['time_range']}.{self.it_number}.emb.npy"
        dir_name = "cache/embedding/"
        full_file_name = dir_name + file_name
        if ut.file_exists(full_file_name) and self.use_cache:
            logger.debug("Loading embedding from cache")
            self._embedding_matrix = np.load(full_file_name)   
        else: 
            logger.debug("Generating embedding")
            self._embedding_matrix = self.embedding_strategy.iterate(self.target_dataset)        
            ut.create_directory_if_not_exists(dir_name)
            np.save(full_file_name, self._embedding_matrix)
        self._embedding_time = time.time() - start
    
    def _cluster(self, min_clusters=3):        
        start = time.time()
        best_silhouete = -2
        kmeans_best_clustering = [0 for _ in range(len(self._embedding_matrix))]
        for number_of_clusters in range(min_clusters, 5+1):
            kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
            kmeans_labels = kmeans.fit_predict(self._embedding_matrix)
            silhouette_avg = silhouette_score(self._embedding_matrix, kmeans_labels)
            if silhouette_avg > best_silhouete:
                kmeans_best_clustering = kmeans_labels
                best_silhouete = silhouette_avg
        self._clustering_time = time.time() - start
        self._clustering_matrix = kmeans_best_clustering
        self._silhouette = best_silhouete
        
    @property
    def silhouette(self):
        self._check_clustering()
        return self._silhouette

    @property
    def embedding_time(self):
        self._check_clustering()
        return self._embedding_time

    @property
    def clustering_time(self):
        self._check_clustering()
        return self._clustering_time    
    
    @property
    def embedding_matrix(self):
        self._check_clustering()
        new_shape = self.target_dataset.shape[1], \
            self.target_dataset.shape[2], self._embedding_matrix.shape[-1]
        return self._embedding_matrix.reshape(new_shape)
        
    @property
    def clustering_matrix(self):
        self._check_clustering()
        new_shape = self.target_dataset.shape[1], \
            self.target_dataset.shape[2]
        return self._clustering_matrix.reshape(new_shape)
           
    def get_total_time(self):
        self._check_clustering()
        return self.embedding_time + self.clustering_time
     
    def _check_clustering(self):
        if not self.clustering_done:
            raise Exception("Error - Clustering has not been performed")

    def save_embedding_matrix(self, base_directory, file_name):
        self._check_clustering()
        np.save(f"{base_directory}/{file_name}.emb.npy", self._embedding_matrix)

    def save_clustering_matrix(self, base_directory, file_name):
        self._check_clustering()
        np.save(f"{base_directory}/{file_name}.npy", self._clustering_matrix)
