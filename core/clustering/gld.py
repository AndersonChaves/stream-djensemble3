import numpy as np
import random
import multiprocessing
import time as libtime
from gldpy import GLD
from core.series_generator import SeriesGenerator
from .IEmbeddingStrategy import IEmbeddingStrategy
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GldStrategy(IEmbeddingStrategy):
    def iterate(self, data: np.ndarray):
        return get_gld_series_representation_from_dataset(data)
    
    def get_description(self):
        raise "gld"

def get_gld_series_representation_from_dataset(target_dataset):
    time, lat, long = target_dataset.shape
    if time == 0:
        return None
    gld_list = np.empty((0, 4), float)

    # Create Series List
    series_list = []
    logger.info("Calculating GLDs...")
    for i in range(lat):
        logger.debug("Calculating GLDs for lat", i)
        for j in range(long):
            cut_start, cut_ending = 0, time
            X, _ = SeriesGenerator().manual_split_series_into_sliding_windows(
                target_dataset[cut_start:cut_ending, i, j], time, n_steps_out=0)
            X = X.reshape((len(X[0])))
            series_list.append(X)

    start = libtime.time()
    gld_list = np.empty((0, 4))
    with multiprocessing.Pool() as pool:
        for result in pool.map(calculate_gld_estimator_using_fmkl, series_list):
            gld_estimators = result
            gld_estimators = np.reshape(gld_estimators, (1, 4))
            gld_list = np.append(gld_list, gld_estimators, axis=0)
    end = libtime.time()
    logger.info("GLDs calculated - Time = ", end-start)
    return gld_list

def calculate_gld_estimator_using_gpd(X: np.array):
    # Call GPD from R -------------------------------
    from rpy2.robjects.packages import importr
    gld = importr('gld')

    from rpy2.robjects import numpy2ri
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter

    # Create a converter that starts with rpy2's default converter
    # to which the numpy conversion rules are added.
    np_cv_rules = default_converter + numpy2ri.converter

    with localconverter(np_cv_rules) as cv:
        #r_series = rpy2.robjects.r(X)
        # todo: gld using r's gpd is not working
        gld_results = gld.fit_gpd(X)
    return gld_results[:, 0]

def calculate_gld_estimator_using_fmkl(X: np.array, use_optimization=False):
    # GLD Using python library
    gld = GLD('FMKL')
    #param_MM = gld.fit_MM(X, [0.5, 1], bins_hist=20, maxiter=1000, maxfun=1000, disp_fit=False)
    indexes = np.array((range(len(X))))
    i = 0
    while (True):
        guess = [random.randint(0, 1) * 0.001, random.randint(0, 1)]
        #guess = (1, 1)
        try:
            param_MM = gld.fit_curve(indexes, X, initial_guess=guess, N_gen=1000,
                                     optimization_phase=use_optimization, shift=True, disp_fit=False)
            return param_MM[0]
        except Exception as e:
            #print(e)
            #print("GLD Error: changing initial guess... (Error) " + str(i))
            i +=1
            continue

def calculate_gld_estimator_using_vsl(X: np.array):
    # GLD Using python library
    gld = GLD('VSL')
    #param_MM = gld.fit_MM(X, [0.5, 1], bins_hist=20, maxiter=1000, maxfun=1000, disp_fit=False)
    indexes = np.array((range(len(X))))
    i = 0
    while (True):
        guess = [random.randint(0, 1) * 0.001, random.randint(0, 1)]
        #guess = (1, 1)
        try:
            param_MM = gld.fit_curve(indexes, X, initial_guess=guess, N_gen=1000,
                                     optimization_phase=False, shift=True, disp_fit=False)
            return param_MM[0]
        except Exception as e:
            #print(e)
            #print("GLD Error: changing initial guess... (Error) " + str(i))
            i += 1
            continue