import numpy as np
import core.utils as ut
from core.dataset_manager import DatasetManager
from clustering.clustering import cluster_dataset, expand_tile
from clustering.clustering import create_yolo_tiling, categorize_dataset

def test_1():
    tiling = np.full((5, 5), -1)
    clustering = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 0, 0],
                           [1, 1, 1, 1, 0],
                           [1, 1, 1, 0, 0],
                           [1, 1, 1, 0, 0]])
    print(expand_tile(tiling, clustering, (0, 0), 1))

def test_2():
    tiling = np.full((5, 5), -1)
    clustering = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 0, 0],
                           [1, 1, 1, 1, 0],
                           [1, 1, 1, 0, 0],
                           [1, 1, 1, 0, 0]])
    print(expand_tile(tiling, clustering, (0, 1), 1))

def test_3():
    tiling = np.full((5, 5), -1)
    clustering = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 0, 0],
                           [1, 1, 1, 1, 0],
                           [1, 1, 1, 0, 0],
                           [1, 0, 1, 0, 0]])
    print(expand_tile(tiling, clustering, (1, 1), 1))

def test_clustering():
    dataset = np.full((10, 5, 5), 7)
    clustering = cluster_dataset(dataset)
    ut.print_array(np.reshape(clustering, (5, 5)))


def test_tiling():
    clustering = np.array([[1, 1, 1, 1, 1],
                           [1, 0, 1, 0, 0],
                           [1, 1, 1, 1, 0],
                           [1, 1, 1, 0, 0],
                           [1, 0, 1, 0, 0]])
    create_yolo_tiling(clustering)

def custom_made_tiling():
    tiling = [[0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1],
              [3, 3, 3, 3, 3, 3, 3],
              [3, 3, 3, 3, 3, 3, 3]]
    tile_dict = {1: {'start': (5, 0), 'end': (6, 6)},
                 2: {'start': (0, 0), 'end': (4, 3)},
                 3: {'start': (0, 4), 'end': (4, 6)}, }
    return tiling, tile_dict

def test_categorization():
    array = DatasetManager().synthetize_dataset(shape=(10, 10, 10))
    print(categorize_dataset(array))

if __name__ == "__main__":
    test_categorization()
    #print_array()
    # test_clustering()
    # print(gld)
