from clustering.static_clustering import StaticClustering
from config.config import Config
import logging
logging.root.setLevel(logging.DEBUG)


if __name__ == "__main__":
    config = Config("config/embedding_evaluation.json")
    for key, cls_configuration in config.data["clustering"].items():
        static_clustering = StaticClustering(cls_configuration)
        static_clustering.run()
        print(static_clustering.get_statistics())
