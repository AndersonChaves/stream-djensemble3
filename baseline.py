import logging
import dataset
import training

logging.root.setLevel(logging.DEBUG)
CONFIG_FILE = "config.json"

if __name__ == "__main__":
    ds = dataset.generate_spatio_temporal_differences_dataset()
    training.train_models(ds)
    

