import requests

# Define the base URL and endpoint
base_url = "http://127.0.0.1:11434/v1"
endpoint = "/chat/completions"

# Define the payload for the API call
payload = {
    "model": "deepseek-r1:1.5b",  # Replace with the model name from `ollama list`
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.8,
    "max_tokens": 16384
}

# Make the API call
try:
    response = requests.post(f"{base_url}{endpoint}", json=payload)
    
    # Check if the response is successful
    if response.status_code == 200:
        result = response.json()
        print("Assistant's response:", result['choices'][0]['message']['content'])
    else:
        print("Error:", response.status_code, response.text)
except Exception as e:
    print("An error occurred:", str(e))
