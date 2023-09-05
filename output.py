from config.config import Config
import sqlite3

config_file = "embedding_evaluation.json"
con = sqlite3.connect("exp1.db")
cur = con.cursor()


def print_parameter_results(embedding_method, parameter):
    res = cur.execute(f"""SELECT AVG({parameter}) FROM exp1 WHERE embedding = '{embedding_method}' """).fetchall()
    print(f"{parameter}: {res[0][0]}")

def print_all_results():
    all_methods = []
    config = Config(f"config/{config_file}")
    for key, cls_configuration in config.data["clustering"].items():
        all_methods .append(cls_configuration['embedding_method'])

    for method in all_methods:
        print(f"Reusults for {method}")
        for parameter in ['embedding', 'numberOfClusters', 'embeddingTime', 'clusteringTime', 'silhouette']:
            print_parameter_results(method, parameter)
        print(f"\n")


print_all_results()
