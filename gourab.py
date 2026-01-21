import csv
import random

def generate_random_numbers_to_csv(filename, num_rows, num_columns, min_val, max_val):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        header = [f'Column_{i+1}' for i in range(num_columns)]
        csv_writer.writerow(header)

        for _ in range(num_rows):
            row = [random.randint(min_val, max_val) for _ in range(num_columns)]
            csv_writer.writerow(row)

output_filename = 'random_numbers.csv'
num_rows_to_generate = 10
num_columns_to_generate = 5
minimum_value = 1
maximum_value = 100

generate_random_numbers_to_csv(output_filename, num_rows_to_generate, num_columns_to_generate, minimum_value, maximum_value)
print(f"Generated '{num_rows_to_generate}' rows and '{num_columns_to_generate}' columns of random numbers in '{output_filename}'")