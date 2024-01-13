import numpy as np
from core.clustering.static_clustering import StaticClustering
from core.clustering.stream_clustering import StreamClustering
from .query_planner import QueryPlanner
from .ensemble_runner import EnsembleRunner
from core.tiling.tiling import Tiling
import time
import logging
import core.view as view
from core.utils import measure_time_decorator

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ContinuousQuery():
    def __init__(self, config: dict):                
        self.config = config                
        self.rmse_history = []
        self.time_history = []
        self.ensemble_history = []

    #----Offline----------------------------------------------------------------
    def prepare(self, pre_clustering_window: np.ndarray, 
                candidate_models):
        if self.config["clustering"]["cluster_query_window"]:
            pre_clustering_window = self._extract_query_window(pre_clustering_window)
        self.planner = QueryPlanner(self.config)
        self.runner = EnsembleRunner(self.config)
        if self.config["clustering"]["type"] in ["static", "dynamic"]:
            self.clustering = StaticClustering(pre_clustering_window, 
                                               self.config["clustering"]["embedding_method"])
            self.clustering.run()
            self.tiling = Tiling(self.config["tiling"], 
                            self.config["query"]["start"])
            self.tiling.run(self.clustering, pre_clustering_window)        
        else:
            self.clustering = StreamClustering(
                self.config["clustering"]["embedding_method"])
            self.clustering.initialize(pre_clustering_window)        
        self.candidate_models = candidate_models

    #----Online-----------------------------------------------------------------
    def run(self, input: np.ndarray, true_output: np.ndarray):
        start_time = time.time()
        self._update_input(input)
        self._update_clustering()
        self._update_tiling()        
        self._define_ensemble()
        self._execute_ensemble()
        self.time_history.append(time.time() - start_time)
        self._update_error(true_output)        

    def _extract_query_window(self, data_window: np.ndarray):
        start = self._get_query_start_after_compacting()
        end = self._get_query_end_after_compacting()

        return data_window[:, start[0]:end[0], start[1]:end[1]]

    def _get_query_start_after_compacting(self):
        start = \
            int(self.config["query"]["start"][0] / self.config["data_source"]["compacting_factor"]),\
            int(self.config["query"]["start"][1] / self.config["data_source"]["compacting_factor"])
        return start
    
    def _get_query_end_after_compacting(self):
        end = \
            int(self.config["query"]["end"][0] / self.config["data_source"]["compacting_factor"]),\
            int(self.config["query"]["end"][1] / self.config["data_source"]["compacting_factor"])

        return end

    @measure_time_decorator
    def _update_input(self, input: np.ndarray):
        if self.config["clustering"]["cluster_query_window"]:
            input = self._extract_query_window(input)
        self.input = input

    @measure_time_decorator
    def _update_clustering(self):
        if self.config["clustering"]["type"] == "stream":
            self.clustering.update(self.input)
        elif self.config["clustering"]["type"] == "dynamic":
            self.clustering = StaticClustering(self.input, 
                self.config["clustering"]["embedding_method"])
            self.clustering.run()
    
    @measure_time_decorator    
    def _update_tiling(self):
        if self.config["clustering"]["type"] in ["dynamic", "stream"]:
            self.tiling = Tiling(self.config["tiling"], 
                self.config["query"]["start"])
            self.tiling.run(self.clustering, self.input)
        logger.debug(f"{self.tiling.get_number_of_tiles()} tiles identified. ")

    @measure_time_decorator
    def _define_ensemble(self):
        self.ensemble = self.planner.define_ensemble(
            self.tiling, self.candidate_models, self.input)

    @measure_time_decorator
    def _execute_ensemble(self):
        self.runner.run(self.ensemble, self.input)
        self.predicted = self.runner.get_prediction()

    #----Query Evaluation------------------------------------------------------------

    def _update_error(self, true_output: np.ndarray):        
        self.true_output = self._extract_query_window(np.reshape(true_output, (1, *true_output.shape)))
        query_rmse = np.average(np.sqrt(np.square(self.predicted - self.true_output)))
        self.rmse_history.append(query_rmse)
        self.ensemble_history.append()
        
    def _generate_visualization(self, true_output, output_path, output_file_name):
        query_area_true_output = self._extract_query_window(
            np.reshape(true_output, (1, *true_output.shape)))
        
        view.save_array_list_as_image([self.predicted[0], query_area_true_output[0]],
            output_path +  "Error/" + output_file_name, title_list=["Predicted", "True"])
        
        view.save_array_list_as_image([self.tiling.tiling_frame],
            output_path + "Tiling/" + output_file_name, title_list=["Tiling"])
        logger.debug(f"***Generated visualization for file {output_file_name}")
        

    def get_last_window_statistics(self):
        self.text = f"Last Error: {self.rmse_history[-1]}\n"
        self.text += \
            f"Average RMSE: {str(sum(self.rmse_history) / len(self.rmse_history))}\n"
        self.text += f"Last window execution time: {self.time_history[-1]}\n"
        return self.text
    
    def get_statistics(self):
        self.text += \
            f"Average RMSE: {str(sum(self.rmse_history) / len(self.rmse_history))}\n"
        self.text += \
            f"Average execution time: {str(sum(self.time_history) / len(self.time_history))}\n"
        return self.text
