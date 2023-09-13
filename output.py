from config.config import Config
import sqlite3

con = sqlite3.connect("exp2.db")
cur = con.cursor()

c_factor = 10
n_clusters = 3
embedding_method= 'parcorr6'

all_configurations = ['A']


def print_parameter_results(configuration, parameter):
    
    res = cur.execute(f"""SELECT parameter) 
                      FROM exp3
                      WHERE configuration = '{configuration}' 
                      """).fetchall()
    parameter_sum = 0
    for row in res:
        row_average = sum(row[0]) / len(row[0])
        parameter_sum += row_average

    print(f"{parameter}: {parameter_sum / len(row)}")

def print_all_results():
    for config in all_configurations:
        print(f"Reusults for {config}")
        for parameter in ['error', 'time']:
            print_parameter_results(config, parameter)
        print(f"\n")


print_all_results()
