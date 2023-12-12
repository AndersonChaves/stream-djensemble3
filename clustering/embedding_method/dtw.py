import numpy as np
import random
import multiprocessing
import time as libtime
from gldpy import GLD
from core.series_generator import SeriesGenerator
from clustering.IEmbeddingStrategy import IEmbeddingStrategy
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DTWStrategy(IEmbeddingStrategy):
    def iterate(self, data: np.ndarray):
        return data
    
    def get_description(self):
        raise "dtw"