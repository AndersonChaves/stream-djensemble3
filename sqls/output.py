import sqlite3

con = sqlite3.connect("exp3.db")
cur = con.cursor()

c_factor = 10
n_clusters = 3
embedding_method= 'parcorr6'
parameters = ['input_data', 'error', 'time']

all_configurations = ['A', "B", "C", "D"]


def print_parameter_results(configuration, parameter):
    
    res = cur.execute(f"""SELECT {parameter}
                      FROM exp3
                      WHERE configuration = '{configuration}' 
                      """).fetchall()
    print(f"Fetched: {len(res)} rows")
    parameter_sum = 0
    param_len = 0
    if parameter in ['error', 'time']:
        for row in res:
            list_of_window_results = [float(x) for x in row[0].strip('][').split(', ')]
            param_len = len(list_of_window_results)
            row_average = sum(list_of_window_results) / param_len
            parameter_sum += row_average
        print(f"{parameter}: {parameter_sum / len(res)}")
    else:
        ch = '\n'
        print(f"{res[0][0].split(ch)}")

def print_all_results():
    for config in all_configurations:
        print(f"Reusults for {config}")
        for parameter in parameters:
            print_parameter_results(config, parameter)
        print(f"\n")


print_all_results()
