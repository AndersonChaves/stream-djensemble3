import pandas as pd
import numpy as np
import metadata
from core.model_training import (
    save_model_as_h5
)
from core.models.lstm_learner import LstmLearner

training_regions = {
    "linear": (0, 0),
    "sinusoidal": (0, 1),
    "random_walk": (0, 3),
    "exponential_growth": (2, 0)
}

def train_models(ds):
    for pattern, region in training_regions.items():
        train_model(ds, pattern, region)     


def train_model(ds, pattern, region):
    bl = metadata.block_length
    time_len = metadata.time_dimension_len

    training_len = int(time_len * 0.9)
    i_start = bl * region[0]
    j_start = bl * region[1]
    training_ds = ds[:training_len, i_start: (i_start+1) *1, 
                        j_start: (j_start+1) *1]
    testing_ds = ds[training_len:, i_start: (i_start+1) *1, 
                        j_start: (j_start+1) *1]
    print("Training Models")
    model = train_lstm(model_name = pattern + str(region), 
                numpy_training_data=training_ds)
    # Test
    print("Predicting Error")
    error = model.predict(testing_ds, series_size=24)

def train_lstm(model_name, numpy_training_data, retrain=False):
    models_dir = "models/"
    model_path = models_dir + model_name
    series_size = 24
    lstm_model = LstmLearner(models_dir, model_name, auto_loading=True)

    if not lstm_model.model_file_exists() or retrain:
        lstm_model.update_architecture(neurons=16, nb_epochs=1,
            batch_size=100, number_of_hidden_layers=2)
        lstm_model.train(numpy_training_data, series_size)
        lstm_model.save()
    return lstm_model
