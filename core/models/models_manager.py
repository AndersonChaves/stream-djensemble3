from core.models.learner import UnidimensionalLearner, MultidimensionalLearner
import os
import core.utils as ut
from core.tiling.tile import Tile
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ModelsManager():
    def __init__(self, config: dict):
        """
            Input: dict config with options:
            {
                convolutional_models_path: <string>,
                temporal_models_path: <string>,
                cost_estimation_function: {
                    noise_level: <float>, 
                    always_update: <bool>
                }
            }
        """
        
        ut.supress_log_messages(True)
        self.temporal_models = []
        self.convolutional_models = []
        self.config = config
        self.conv_models_path = config["convolutional_models_path"]
        self.temp_models_path = config.get("temporal_models_path", "")
        self.include_convolutional_models_from_directory(self.conv_models_path)
        self.include_temporal_models_from_directory(self.temp_models_path)
        self.load_models()
        ut.supress_log_messages(False)

    def load_models(self):
        logger.debug("Loading temporal models")
        self.temporal_models      = self.load_all_temporal_models()
        logger.debug("Loading convolutional models")
        self.convolutional_models = self.load_all_convolutional_models() 
        
    def load_all_temporal_models(self):
        model_list = []
        if self.temp_models_path == "":
            return []
        for file in os.listdir(self.temp_models_path):
            if file.endswith(".h5"):
                model_name = file[:-3]
                model_list.append(self.load_temporal_model(model_name))
        return model_list

    def load_all_convolutional_models(self):
        models_names = ut.get_names_of_models_in_dir(self.conv_models_path)
        models_list = [self.load_convolutional_model(model) for model in models_names]
        return models_list

    def load_model(self, model_name):
        if model_name.startswith('best_model'):
            return self.load_temporal_model(model_name)
        else:
            return self.load_convolutional_model(model_name)

    def load_temporal_model(self, model_name):
        return UnidimensionalLearner(self.temp_models_path,
                                     model_name,
                                     is_temporal_model=True)

    def load_convolutional_model(self, model_name):
        return MultidimensionalLearner(self.convolutional_models_directory, model_name, is_temporal_model=False)

    def include_convolutional_models_from_directory(self, models_path):
        self.convolutional_models_directory = models_path

    def include_temporal_models_from_directory(self, models_path):
        self.temp_models_path = models_path

    def get_latitude_input_size(self, model_name):
        model = self.get_model_from_name(model_name)
        if model.is_temporal_model:
            return 1
        else:
            return int(model.get_model().input.shape[2])

    def get_longitude_input_size(self, model_name):
        model = self.get_model_from_name(model_name)
        if model.is_temporal_model:
            return 1
        else:
            return int(model.get_model().input.shape[3])

    def get_model_from_name(self, model_name):
        all_learners = self.temporal_models + self.convolutional_models
        if all_learners == []:
            logger.debug("No models loaded. Inform models directory and run function load_models")
            raise(Exception)
        for learner in all_learners:
            name = learner.get_name()
            if name == model_name:
                return learner
        logger.debug("Model not found")
        raise(Exception)

    def get_models_from_list_of_names(self, best_allocation):
        models_list = []
        for model_name in best_allocation:
            models_list.append(self.get_model_from_name(model_name))
        return models_list

    def get_models(self):
        return self.temporal_models + self.convolutional_models

    def get_error_estimative_ranking(self, dataset,
                                       tile: Tile, dataset_manager, candidate_models):
        error_estimative = {}
        data_from_tile_region = dataset_manager.get_data_from_tile(dataset, tile)
        for learner in candidate_models:
            #if learner.name == "CFSR-2014.nc-x0=(80, 9)-7x7-(\'2014-01-01 00:00:00\', \'2014-03-30 23:45:00\')-summer"

            error_est = learner.execute_eef(data_from_tile_region, tile)
            #error_est = error_est[0][0]
            if error_est < 0:
                raise(Exception("Error - Cef Value is negative"))
            #error_est = learner.execute_eef(data_from_tile_region, tile)
            error_estimative[learner.name] = error_est
        return error_estimative

    def get_models_from_list(self, models_list):
        models = []
        for model_name in models_list:
            models.append(self.get_model_from_name(model_name))
        return models

    def update_cef(self):
        ut.supress_log_messages(True)
        noise_level_for_cef = self.config["cost_estimation_function"]["noise_level"]
        update_models_cef = self.config["cost_estimation_function"]["always_update"] == True
        for learner in self.get_models():
            logger.debug(f"Updating CEF for model {learner}")
            learner.update_cef(noise_level_for_cef, update_models_cef=update_models_cef)
        ut.supress_log_messages(False)