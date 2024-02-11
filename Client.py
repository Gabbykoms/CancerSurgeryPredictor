import requests
import pandas as pd

# Example input data
df = pd.read_csv('BRCA.csv')

# Select a single row from the dataset for prediction (assuming it's the first row)
data = df.iloc[0].to_json()

# Send POST request to the Flask endpoint
response = requests.post('http://127.0.0.1:5000/predict', json=data)

# Get the prediction from the response
prediction = response.json()['prediction']
print('Prediction:', prediction)
