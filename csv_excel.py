import pandas as pd

# Load CSV from same folder as script
df = pd.read_csv('study_data.csv')  # Replace 'data.csv' with your file name
print(df.head())  # View first 5 rows
print(df.shape)   # Rows and columns count
