from clustering.static_clustering import StaticClustering
from config.config import Config
import logging
from tinydb import TinyDB, Query

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


config_file = "embedding_evaluation.json"
number_of_runs = 2

def run_experiment(config, it_number):    
    for key, cls_configuration in config.data["clustering"].items():
        static_clustering = StaticClustering(cls_configuration)
        static_clustering.set_cache(it_number)
        static_clustering.run()
        file_name = f"{cls_configuration['embedding_method']}t{cls_configuration['data_source']['time_range']}"
        static_clustering.save_clustering_image(f"output/images/{it_number}", file_name)
        
        with open(f'r{it_number+1}.out', 'w') as f:
            f.write(static_clustering.get_statistics())


if __name__ == "__main__":    
    config = Config(f"config/{config_file}")
    print(f"Config file{config_file}")
    for run in range(number_of_runs):
        print(f"Iteration:{run}")
        run_experiment(config, run)
    
