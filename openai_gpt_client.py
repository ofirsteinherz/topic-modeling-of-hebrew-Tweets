import os
import requests
import json

class OpenAIGPTClient:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key is not set. Please set the OPENAI_API_KEY environment variable.")
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def llm_call(self, messages, model="gpt-4o-mini", temperature=0.3, max_tokens=500, json_format=False, seed=42):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed
        }

        if json_format:
            data["response_format"] = { "type": "json_object" }

        try:
            response = requests.post(self.endpoint, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raise HTTPError for bad responses
            return response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.HTTPError as http_err:
            error_message = response.json().get("error", {}).get("message", "")
            return f"HTTP error occurred: {http_err}\nDetails: {error_message}"
        except Exception as err:
            return f"An error occurred: {err}"
        
# # Example usage of `OpenAIGPTClient`
# client = OpenAIGPTClient()
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#     {"role": "user", "content": "Where was it played?"}
# ]
# response = client.llm_call(messages, max_tokens=50)
# print(response)