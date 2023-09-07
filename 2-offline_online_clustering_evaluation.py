from clustering.static_clustering import StaticClustering
from clustering.clustering_through_time import ClusteringThroughTime
from config.config import Config
import logging
import sqlite3

logging.root.setLevel(logging.INFO)

def perform_experiment(configuration, it_number):
    print(f"Performing Clustering: Configuration {key}")
    cls_through_time = ClusteringThroughTime(configuration)
    print(cls_through_time.get_parameters())
    cls_through_time.run()
    print(cls_through_time.get_statistics())
    print("---"*10)    

    with open(f'r{it_number+1}.out', 'a') as f:
            f.write(cls_through_time.get_statistics())

    con = sqlite3.connect("exp2.db")
    cur = con.cursor()
    try:
        cur.execute("""
            CREATE TABLE exp2( \
                    iteration, type, embedding, \
                    numberOfClusters, total_time, \
                    avgSilhouette, avgSilhouetteSecondHalf, 
                    silhouetteHistory, compacting_factor)
        """)
    except:
        print("Could not create table")

    cur.execute(f"""
            INSERT INTO exp2 VALUES ( 
            {it_number}, 
            '{configuration["type"]}', 
            '{cls_through_time.embedding_method}', 
            {cls_through_time.n_clusters[0]}, 
            {cls_through_time.time_end - cls_through_time.time_start}, 
            {cls_through_time.average_silhouette},
            {cls_through_time.average_silhouette_2nd_half},
            '{cls_through_time.silhouette_history}', 
            {configuration["clustering"]["data_source"]["compacting_factor"]}
            )"""
    )        
    con.commit()        

if __name__ == "__main__":
    config = Config("config/config-parcorr6.json")
    for key, configuration in config.data["clustering_through_time"].items():
        if key in config.data["skip_list"]:
            continue
        for gconfig, value in config.data["global_configuration"].items():
            configuration[gconfig] = value
        for it_number in range(2):
            perform_experiment(configuration, it_number+1)