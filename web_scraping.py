import pandas as pd

url = "https://www.w3schools.com/html/html_tables.asp"

tables = pd.read_html(url)      
df = tables[0]            
print(df.head())

subset = df.head(5)
subset.to_csv("web_table.csv",index=False)