from clustering.static_clustering import StaticClustering
from config.config import Config
import logging
logging.root.setLevel(logging.DEBUG)


if __name__ == "__main__":
    config = Config("config/embedding_evaluation.json")
    for key, cls_configuration in config.data["clustering"].items():
        static_clustering = StaticClustering(cls_configuration)
        static_clustering.run()
        file_name = f"{cls_configuration['embedding_method']}t{cls_configuration['data_source']['time_range']}"
        static_clustering.save_clustering_image("output/images/", file_name)
        #static_clustering.save_clustering_matrix("embedding/", file_name)
        print(static_clustering.get_statistics())
