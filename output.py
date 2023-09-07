from config.config import Config
import sqlite3

con = sqlite3.connect("exp2.db")
cur = con.cursor()

c_factor = 10
n_clusters = 3
embedding_method= 'parcorr6'

all_types = ['static', 'dynamic', 'stream']


def print_parameter_results(clustering_type, parameter):
    res = cur.execute(f"""SELECT AVG({parameter}) 
                      FROM exp2 
                      WHERE embedding = '{embedding_method}' 
                      AND compacting_factor == {c_factor}
                      AND numberOfClusters = {n_clusters}""").fetchall()
    print(f"{parameter}: {res[0][0]}")

def print_all_results():
    for type in all_types:
        print(f"Reusults for {type}")
        for parameter in ['embedding', 'numberOfClusters', 'total_time', 
                          'avgSilhouette', 'avgSilhouetteSecondHalf']:
            print_parameter_results(type, parameter)
        print(f"\n")


print_all_results()
