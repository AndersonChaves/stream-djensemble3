import numpy as np
import multiprocessing
from core.series_generator import SeriesGenerator
from itertools import repeat
import logging
from clustering.IEmbeddingStrategy import IEmbeddingStrategy

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ParcorrStrategy(IEmbeddingStrategy):
    def __init__(self, **params):
        self.basis_size = params["basis_size"]
        self.parcorr_basis = generate_parcorr_random_vectors(
            vector_size=params["series_size"], basis_size=self.basis_size)

    def iterate(self, data: np.ndarray):
        series_size, _, _ = data.shape
        parcorr_embedding = get_parcorr_series_representation_from_dataset(
            data, self.basis_size, self.parcorr_basis)
        return parcorr_embedding
    
    def get_description(self):
        raise "parcorr" + str(self.basis_size)

def get_parcorr_series_representation_from_dataset(
        target_dataset: np.array, basis_size: int, parrcorr_basis=None):
    time, lat, long = target_dataset.shape
    if time == 0:
        return None
    series_size = time
    if parrcorr_basis is None:
        parrcorr_basis = generate_parcorr_random_vectors(vector_size=series_size, basis_size=basis_size)

    # Create Series List
    series_list = []
    for i in range(lat):
        # print("Calculating parcorr embedding for lat", i)
        for j in range(long):
            cut_start, cut_ending = 0, time
            X, _ = SeriesGenerator().manual_split_series_into_sliding_windows(
                target_dataset[cut_start:cut_ending, i, j], time, n_steps_out=0)
            X = X.reshape((len(X[0])))
            series_list.append(X)

    sketches_list = np.empty((0, basis_size))
    list_parcorr_basis = list(repeat(parrcorr_basis, len(series_list)))
    with multiprocessing.Pool() as pool:
        for sketch in pool.starmap(calculate_vector_sketch,
                                   zip(series_list, list_parcorr_basis)):
            sketches_list = np.append(sketches_list, np.expand_dims(sketch, axis=0), axis=0)
    logger.info("Vector sketches calculated")
    return sketches_list

def calculate_vector_sketch(vector, basis):
    sketch = []
    for b in basis:
        sketch.append(np.dot(b, vector))
    return sketch

def generate_parcorr_random_vectors(vector_size, basis_size):
    basis = []
    for _ in range(basis_size):
        basis.append((np.random.rand(1,vector_size)[0] - 0.5) * 2)
    return basis
