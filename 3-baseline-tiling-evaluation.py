import core.djensemble 
from core.config import Config
import logging
import sqlite3
import core.utils as ut
import os, shutil

logging.root.setLevel(logging.DEBUG)
CONFIG_FILE = "config-baseline.json"

WRITE_TO_FILE = True
WRITE_TO_DATABASE = True
MAX_ITERATIONS = 1


def save_in_database(configuration, it_number, djensemble, description=""):
    print("Writing in files")
    if WRITE_TO_FILE:
        with open(f'tmp/r_{description}_{it_number+1}.out', 'w') as f:
            f.write(djensemble.get_parameters())
            f.write(djensemble.get_statistics())

    print("Writing to database")
    if not WRITE_TO_DATABASE:
        return
    con = sqlite3.connect("exp3-baseline-temporal.db")
    cur = con.cursor()
    try:
        cur.execute("""
            CREATE TABLE exp3( \
                    iteration, configuration, input_data, \
                    description, error, average_error, time, average_time, 
                    CONSTRAINT exp3_p_key PRIMARY KEY (iteration, configuration, description)
            )            
        """)

    except:
        print("Could not create table")

    try:
        cur.execute(f"""
                INSERT INTO exp3 VALUES ( 
                {it_number}, 
                '{configuration["config"]}', 
                '{djensemble.get_parameters()}',
                '{description.replace("'", "*")}',
                '{djensemble.error_history()}', 
                {sum(djensemble.error_history())/len(djensemble.error_history())}, 
                '{djensemble.time_history()}',
                {sum(djensemble.time_history())/len(djensemble.time_history())} 
                )"""
        )        
        con.commit()        
    except Exception as e:
        print(f"Could not insert into database iteration {it_number}")
        print(e)

def perform_experiment(configuration, it_number, description):
    djensemble = core.djensemble.DJEnsemble(configuration)
    print(djensemble.get_parameters())
    djensemble.run()
    print(djensemble.get_statistics())
    print("---"*10)    
    save_in_database(configuration, it_number, djensemble, description)


def create_directory_with_single_model(model, dir):
    files_list = ut.list_all_files_in_dir(dir, prefix=model)
    if not os.path.exists("tmp/"):
        os.makedirs("tmp/")

    for file in files_list:
        shutil.copy2(dir + file, "tmp/" + file)
    return "tmp/"
    

def perform_single_model_experiment(configuration, it_number):
    temporal_models_path = configuration["models"]["temporal_models_path"]
    convolutional_models_path = configuration["models"]["convolutional_models_path"]
    
    temporal_models = ut.get_names_of_models_in_dir(temporal_models_path)
    convolutional_models = ut.get_names_of_models_in_dir(convolutional_models_path)

    for model in temporal_models:
        temp_directory = create_directory_with_single_model(model, temporal_models_path)
        configuration["models"]["temporal_models_path"] = temp_directory
        configuration["models"]["convolutional_models_path"] = "/"
        perform_experiment(configuration, it_number, description=model)
        ut.remove_directory(temp_directory)

    # for model in convolutional_models:
    #     temp_directory = create_directory_with_single_model(model, convolutional_models_path)
    #     configuration["models"]["temporal_models_path"] = "/"
    #     configuration["models"]["convolutional_models_path"] = temp_directory
    #     perform_experiment(configuration, it_number, description=model)
    #     ut.remove_directory(temp_directory)
        

if __name__ == "__main__":
    config = Config(f"resources/config/{CONFIG_FILE}")
    for key, configuration in config.data["djensemble"].items():
       if key in config.data["skip_list"]:
           print(f"Skipping Configuration {key}")  
           continue
       else:
           print(f"Performing DJEnsemble: Configuration {key}")
       configuration["config"] = key
       for gconfig, value in config.data["global_configuration"].items():
           configuration[gconfig] = value
       for it_number in range(0, MAX_ITERATIONS):
           perform_single_model_experiment(configuration, it_number)       
