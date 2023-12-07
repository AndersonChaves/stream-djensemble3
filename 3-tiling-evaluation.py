import core.djensemble 
from core.config import Config
import logging
import sqlite3

logging.root.setLevel(logging.DEBUG)
CONFIG_FILE = "config.json"

WRITE_TO_FILE = True
WRITE_TO_DATABASE = True
MAX_ITERATIONS = 5


def save_in_database(configuration, it_number, djensemble):
    print("Writing in files")
    if WRITE_TO_FILE:
        with open(f'output/files/r{it_number+1}-{configuration["config"]}.out', 'w') as f:
            f.write(djensemble.get_parameters())
            f.write(djensemble.get_statistics())

    print("Writing to database")
    if not WRITE_TO_DATABASE:
        return
    con = sqlite3.connect("exp3.db")
    cur = con.cursor()
    try:
        cur.execute("""
            CREATE TABLE exp3( \
                    iteration, configuration, input_data, \
                    error, average_error, time, average_time, ensemble_history, n_tiles,
                    CONSTRAINT exp3_p_key PRIMARY KEY (iteration, configuration)
            )            
        """)

    except:
        print("Could not create table")

    try:
        print(djensemble.ensemble_history())

        query = f"""
                INSERT INTO exp3 VALUES ( 
                {it_number}, 
                '{configuration["config"]}', 
                '{djensemble.get_parameters()}',
                '{djensemble.error_history()}', 
                {sum(djensemble.error_history())/len(djensemble.error_history())}, 
                '{djensemble.time_history()}',
                {sum(djensemble.time_history())/len(djensemble.time_history())},
                '{str(djensemble.ensemble_history()).replace("'", "*")}',
                '{djensemble.number_of_tiles_history()}'
                )"""
        cur.execute(query)        
        con.commit()        
    except Exception as e:
        print(f"Could not insert into database iteration {it_number}")
        print(e)

def perform_experiment(configuration, it_number):
    djensemble = core.djensemble.DJEnsemble(configuration)
    print(djensemble.get_parameters())
    djensemble.run()
    print(djensemble.get_statistics())
    print("---"*10)    

    save_in_database(configuration, it_number, djensemble)

if __name__ == "__main__":
    config = Config(f"resources/config/{CONFIG_FILE}")
    for key, configuration in config.data["djensemble"].items():
       print(f"Performing DJEnsemble: Configuration {key}")
       configuration["config"] = key
       for gconfig, value in config.data["global_configuration"].items():
            configuration[gconfig] = value
       for it_number in range(0, MAX_ITERATIONS):
            perform_experiment(configuration, it_number)       
