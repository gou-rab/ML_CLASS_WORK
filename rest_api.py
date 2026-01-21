import requests
import pandas as pd

# Example: Fetch JSON data from public API
url = 'https://jsonplaceholder.typicode.com/users'  # Free test API
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
    print(df.head())
    df.to_csv('api_data.csv', index=False)  # Save to CSV
else:
    print(f"Error: {response.status_code}")
