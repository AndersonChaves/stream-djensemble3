from core.ensemble import Ensemble
from models.models_manager import ModelsManager
from models.learner import UnidimensionalLearner
from tiling.tiling import Tile
from unittest.mock import Mock
from core.config import Config
import numpy as np
import sys, os

def create_block_time_series_dataset(dim):
    series_block_1 = np.array(
        ([[[20] * 5] * dim] + [[[25] * 5] * dim]) * dim
    )
    series_block_2 = np.zeros((10, dim, dim))

    series_block_3 = np.array(
        [[[x]*dim]*dim for x in range(-9, 1)]
    )
    
    series_block_4 = np.zeros((10, dim, dim))
    
    row_1 = np.concatenate(
        (series_block_1, series_block_2), axis=1
    )
    row_2 = np.concatenate(
        (series_block_3, series_block_4), axis=1
    )
    
    return np.concatenate((row_1, row_2), axis=2)


def create_noise_time_series_dataset(shape, noise=0):
    time, lat, long = shape

    series = []
    for i in range(lat // 3 * long):
        series.append([k for k in range(time)])
    for i in range(lat // 3 * long):
        series.append([3 - (k % 3) for k in range(time)])
    for i in range((lat // 3 + lat % 3) * long):
        series.append([10 for _ in range(time)])

    array = np.reshape(np.array(series), (lat, long, time))
    array = np.swapaxes(array, 0, 2)

    if noise > 0:
        noise = np.random.normal(0, noise, array.shape)
        array = array + noise
    return array

def create_mock_ensemble():
    config = Config("config/config-tests.json")
    tests_config = config.data["tests"]["runner_test"]
    tiles = []
    tiles += [Tile("1", coordinates=((0, 0), (5, 5)), 
            centroid_coordinates=(2, 2), 
            centroid_series=np.array([1, 2, 3, 4, 5]))]
    
    tiles += [Tile("2", coordinates=((5, 0), (10, 5)), 
            centroid_coordinates=(5, 0), 
            centroid_series=np.array([1, 2, 3, 4, 5]))]
    
    tiles += [Tile("3", coordinates=((0, 5), (5, 10)), 
            centroid_coordinates=(0, 5), 
            centroid_series=np.array([1, 2, 3, 4, 5]))]
    
    tiles += [Tile("4", coordinates=((5, 5), (10, 10)), 
            centroid_coordinates=(5, 5), 
            centroid_series=np.array([1, 2, 3, 4, 5]))]
    
    model = UnidimensionalLearner(tests_config["temporal_model_path"],
                                  tests_config["temporal_model_name"],
                                  is_temporal_model=True)
    ensemble = Ensemble()
    for tile in tiles:
        ensemble.add_item(tile, model, 0.1)
    return ensemble