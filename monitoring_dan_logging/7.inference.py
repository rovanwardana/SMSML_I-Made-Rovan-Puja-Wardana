import requests
import json

EXPORTER_URL = "http://127.0.0.1:8000/predict"

payload = {
    "inputs": [[
        0,
        40.0,
        0,
        0,
        0,
        10.98,
        8.8,
        280
    ]]
}

print("Sending prediction request...")
response = requests.post(
    EXPORTER_URL,
    json=payload,
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    result = response.json()
    prediction = result['predictions'][0]
    print(f"Prediction: {prediction}")
    print("Result: Diabetes" if prediction == 1 else "Result: No Diabetes")
else:
    print(f"Error: {response.status_code}")
    print(response.text)