import numpy as np
import core.utils as ut
from clustering.embedding_method.gld import *
from clustering.embedding_method.parcorr import *
from clustering.embedding_method.dtw import *

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_embedding_strategy(data, embedding_strategy_name):
    if embedding_strategy_name == "gld":
        return GldStrategy()
    elif embedding_strategy_name.startswith("parcorr"):
        basis_size = int(embedding_strategy_name[7:])        
        return ParcorrStrategy(basis_size=basis_size, series_size = data.shape[0])
    elif embedding_strategy_name.startswith("dtw"):
        return DTWStrategy()
    else:
        raise Exception("Clustering Method chosen has not been implemented")

def get_embedded_series_representation(target_dataset, method):
    if method == "gld":
        series_embedding_matrix = get_gld_series_representation_from_dataset(target_dataset)
        normalize_embedding_list(series_embedding_matrix)
    elif method.startswith("parcorr"):
        basis_size = int(method[7:])
        series_embedding_matrix = get_parcorr_series_representation_from_dataset(
            target_dataset, basis_size)
    else:
        raise Exception("Clustering Method chosen has not been implemented")
    return series_embedding_matrix


def load_embedding_if_exists(base_directory,
                             dataset_name,
                             embedding_method,
                             time_start, time_end):
    file_name = base_directory + dataset_name + "-" + \
        embedding_method + "-" + \
        "time" + str(time_start) + "to" + str(time_end) + \
        ".embedding.npy"
    embedding = None
    print("Trying to load ", file_name)
    if ut.file_exists(file_name):
        embedding = np.load(file_name)
        print("Load successful")
    return embedding

def save_embedding(embedding, base_directory, dataset_name, embedding_method, time_start, time_end):
    file_name = base_directory + dataset_name + "-" + \
        embedding_method + "-" + \
        "time" + str(time_start) + "to" + str(time_end) + \
        ".embedding.npy"
    np.save(file_name, embedding)



def normalize_embedding_list(gld_list):
    for att in range(len(gld_list[0])):
        values_per_attribute = gld_list[:, att]
        min_val = min(values_per_attribute)
        max_value = max(values_per_attribute)
        normalized = [(x - min_val) / (max_value - min_val) * 100 for x in values_per_attribute]
        gld_list[:, att] = normalized
    return gld_list

