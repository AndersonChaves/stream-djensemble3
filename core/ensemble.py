import numpy as np
from core.models.learner import Learner
import logging
from core.tiling.tile import Tile
import sys, os

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Ensemble():
    def __init__(self):        
        self.tile_data = {}
        self.ensemble_built = False

    def add_item(self, tile, model, error_estimative):
        self.tile_data[tile] = {"best_model": model, 
                                "best_error": error_estimative}        

    def run(self, target_dataset):   
        logger.info("Performing Predictions...")
        i = 0
        for tile, tile_data in self.tile_data.items():
            i += 1
            logger.debug(f"Tile {i} of {len(self.tile_data.keys())}")
            tile_data["prediction"] = self._get_tile_prediction(
                tile, target_dataset)
            logger.debug(f"--->Model {self.tile_data[tile]['best_model'].model_name}")
        self.ensemble_built = True
        
    def get_tiles(self):
        return [tile for tile in self.tile_data.keys()]

    def get_ensemble(self):
        ensemble = [(tile.id, self.tile_data[tile]["best_model"].model_name.replace("'", "*").replace(",", "*")) 
                  for tile in  self.get_tiles()]        
        return ensemble
    

    def get_number_of_models(self):
        list_of_models = []
        for _, model in self.get_ensemble():
            if model not in list_of_models:
                list_of_models.append(model)
        return len(list(set(list_of_models)))

    def get_tile_predicted_frame(self, tile):
        self._check_ensemble_built()
        return self.tile_data[tile]["prediction"]

    def _get_tile_prediction(self, tile, target_dataset):        
        self._supress_log_messages("on")
        model = self.tile_data[tile]["best_model"]
        prediction = self._perform_prediction(target_dataset, model, tile)        
        self._supress_log_messages("off")
        return prediction
    

    def _perform_prediction(self, data_window: np.array, 
                           learner: Learner, tile: Tile):
        start = tile.start_coord
        end = tile.end_coord
        input_dataset = data_window[:, start[0]: end[0]+1, 
                                    start[1]:end[1]+1]
        return learner.invoke_on_dataset(input_dataset)    
    
    def _check_ensemble_built(self):
        assert(self.ensemble_built)

    def _supress_log_messages(self, switch:str):
        if switch == "on":
            self.original_stdout = sys.stdout
            null_device = open(os.devnull, 'w')
            sys.stdout = null_device
        else:
            sys.stdout = self.original_stdout