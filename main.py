import core.djensemble 
from core.config import Config
import logging
import sqlite3
from format.config_parser import parse_dj_configurations
from format.persist import save_in_database

CONFIG_FILE = "input/config/config.json"
MAX_ITERATIONS = 2

def run_configuration(configuration, iteration, database_file):
    djensemble = core.djensemble.DJEnsemble(configuration)
    print(djensemble.get_parameters())
    djensemble.run()
    print(djensemble.get_statistics())
    print("---"*10)
    save_in_database(configuration, iteration, djensemble, database_file)


def perform_experiment(config: Config):
    config = config.data
    if "queries" in config.keys():
        queries = Config(config["queries"]).data["queries"]
    for query_key, query in queries.items():
        for dj_key, _ in config["djensemble"].items():
            if dj_key in config["skip_list"]:
                continue
            configuration = parse_dj_configurations(query, config, dj_key)
            for it in range(MAX_ITERATIONS):
                database_file = f"{query_key}.db"
                run_configuration(configuration, it, database_file)

if __name__ == "__main__":        
    perform_experiment(Config(CONFIG_FILE))