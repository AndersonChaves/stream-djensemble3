import csv
import re
import sys

csv.field_size_limit(sys.maxsize)
# Function to extract list of values from a string
def extract_list_of_values(string):
    # Extracting numbers from the string using regular expression
    numbers = re.findall(r'[\d,]+', string)
    # Converting extracted numbers to floats
    #numbers = [float(num.replace(',', '.')) for num in numbers]
    return numbers

output_csv_file = "temp_output.csv"

# Path to your CSV file
csv_file = 'results-08022024-modified.csv'

# Open the CSV file
with open(csv_file, newline='') as file:
    # Create a CSV reader object
    reader = csv.DictReader(file, delimiter=';')

    extracted_error_values = []
    # Iterate through each row in the CSV file
    for row in reader:
        # Extract the list of values from the 'error' column
        error_values = extract_list_of_values(row['error'])

        extracted_error_values.append(error_values)


        # Print the extracted list of values
        #print('List of values:')
        #_ = [print(x) for x in error_values]
        #a = input()

transposed_extracted_error_values = list(map(list, zip(*extracted_error_values)))

# Open the output CSV file for writing
with open(output_csv_file, 'w', newline='') as output_file:
    # Create a CSV writer object
    writer = csv.writer(output_file)

    # Write the transposed extracted error values to the output CSV file
    writer.writerows(transposed_extracted_error_values)











