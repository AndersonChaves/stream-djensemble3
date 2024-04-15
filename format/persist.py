import sqlite3

def save_in_database(configuration, it_number, djensemble, database_file):
    save_raw_data(configuration, it_number, djensemble, database_file)        

def save_raw_data(configuration, it_number, djensemble, database_file):
    print("Writing to database")
    con = sqlite3.connect(database_file)
    cur = con.cursor()
    try:
        cur.execute("""
            CREATE TABLE exp( \
                    iteration, 
                    configuration,
                    window, 
                    input_data, \
                    error, 
                    time, 
                    ensemble, 
                    n_tiles,
                    CONSTRAINT exp3_p_key PRIMARY KEY (iteration, configuration, window)
            )
        """)

    except:
        print("Could not create table")

    try:
        print(djensemble.ensemble_history())

        for window in range(len(djensemble.error_history())):
            query = f"""
                    INSERT INTO exp VALUES ( 
                    {it_number}, 
                    '{configuration["config"]}', 
                    {window+1}, 
                    '{djensemble.get_parameters()}',
                    '{djensemble.error_history()[window]}',                    
                    '{djensemble.time_history()[window]}',                    
                    '{str(djensemble.ensemble_history()[window]).replace("'", "*")}',
                    '{djensemble.number_of_tiles_history()[window]}'                    
                    )"""
            cur.execute(query)        
            con.commit()        
    except Exception as e:
        print(f"Could not insert into database iteration {it_number}")
        print(e)


def update_winners(database):
    # Add a new column "winner_error"
    add_column_if_not_exists(database, 'exp', 'winner_error')
    add_column_if_not_exists(database, 'exp', 'winner_time')
    add_column_if_not_exists(database, 'exp', 'winner_n_tiles')
    
    # Connect to the SQLite database
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    
    # Select minimum error for each iteration and window combination
    cur.execute("""
    UPDATE exp 
    SET winner_error = (CASE 
                            WHEN error = (
                                SELECT MIN(error) 
                                FROM exp AS e 
                                WHERE e.iteration = exp.iteration 
                                    AND e.window = exp.window
                            ) THEN 1 
                            ELSE 0 
                        END)
    """)

    # Select minimum time for each iteration and window combination
    cur.execute("""
    UPDATE exp 
    SET winner_time = (CASE 
                            WHEN time = (
                                SELECT MIN(time) 
                                FROM exp AS e 
                                WHERE e.iteration = exp.iteration 
                                    AND e.window = exp.window
                            ) THEN 1 
                            ELSE 0 
                        END)
    """)

    # Select minimum error for each iteration and window combination
    cur.execute("""
    UPDATE exp 
    SET winner_n_tiles = (CASE 
                            WHEN n_tiles = (
                                SELECT MIN(n_tiles) 
                                FROM exp AS e 
                                WHERE e.iteration = exp.iteration 
                                    AND e.window = exp.window
                            ) THEN 1 
                            ELSE 0 
                        END)
    """)

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

def add_column_if_not_exists(database, table, target_column):    
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    columns = cur.fetchall()
    if not any(column[1] == target_column for column in columns):
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {target_column}")
    conn.commit()
    conn.close()

def save_average_all_iterations(database_file):
    # Connect to the database
    conn = sqlite3.connect(database_file)
    cur = conn.cursor()

    try:
        cur.execute("""
        CREATE TABLE exp_condensed(
                            iteration, 
                            configuration, 
                            average_error, 
                            average_time, 
                            average_n_tiles
                    )
        """)
    except:
        print("Could not create table exp_condensed")

    # Populate exp_condensed table with average values
    cur.execute("""
    INSERT INTO exp_condensed (iteration, configuration, average_error, average_time, average_n_tiles)
    SELECT iteration, configuration, AVG(error) as average_error, AVG(time) as average_time, AVG(n_tiles) as average_n_tiles
    FROM exp
    GROUP BY iteration, configuration
    """)

    # Commit changes and close connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    database_file = "Q1.db"
    update_winners(database_file)
    save_average_all_iterations(database_file)
    