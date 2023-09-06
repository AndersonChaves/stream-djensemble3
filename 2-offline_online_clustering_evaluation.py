from clustering.static_clustering import StaticClustering
from clustering.clustering_through_time import ClusteringThroughTime
from config.config import Config
import logging

logging.root.setLevel(logging.ERROR)

def perform_experiment(configuration):
    print(f"Performing Clustering: Configuration {key}")
    cls_through_time = ClusteringThroughTime(configuration)
    print(cls_through_time.get_parameters())
    cls_through_time.run()
    print(cls_through_time.get_statistics())
    print("---"*10)    

if __name__ == "__main__":
    config = Config("config/config-parcorr6.json")
    for key, configuration in config.data["clustering_through_time"].items():
        if key in config.data["skip_list"]:
            continue
        perform_experiment(configuration)