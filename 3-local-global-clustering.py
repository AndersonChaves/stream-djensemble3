import core.djensemble 
from config.config import Config
import logging
logging.root.setLevel(logging.INFO)

def perform_experiment(configuration):
    djensemble = core.djensemble.DJEnsemble(configuration)
    print(djensemble.get_parameters())
    djensemble.run()
    print(djensemble.get_statistics())
    print("---"*10)    

if __name__ == "__main__":
    config = Config("config/config.json")
    for key, configuration in config.data["djensemble"].items():
       print(f"Performing DJEnsemble: Configuration {key}")
       perform_experiment(configuration)        
