import sqlite3

con = sqlite3.connect("exp3.db")
cur = con.cursor()

c_factor = 10
n_clusters = 3
embedding_method= 'parcorr6'

all_configurations = ['A']


def print_parameter_results(configuration, parameter):
    
    res = cur.execute(f"""SELECT {parameter}
                      FROM exp3
                      WHERE configuration = '{configuration}' 
                      """).fetchall()
    #print("Res is: ", res)
    parameter_sum = 0
    param_len = 0
    for row in res:
        param_list = [float(x) for x in row[0].strip('][').split(', ')]
        param_len = len([float(x) for x in row[0].strip('][').split(', ')])
        row_average = sum(param_list) / len(param_list)
        parameter_sum += row_average
    print(f"{parameter}: {parameter_sum / param_len}")

def print_all_results():
    for config in all_configurations:
        print(f"Reusults for {config}")
        for parameter in ['error', 'time']:
            print_parameter_results(config, parameter)
        print(f"\n")


print_all_results()
