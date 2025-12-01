
import requests
from openai import OpenAI

class OllamaAPI():
    def __init__(self, endpoint:str, token:str):
        self.endpoint = endpoint
        self.token = token

    def generate(self, model:str, prompt:str):
        response = requests.post(self.endpoint, 
                                 headers={"Authorization": f"Bearer {self.token}"},
                                 json = {
                                     "model":model,
                                     "prompt": prompt,
                                     "stream": False
                                 })
        response.raise_for_status()

        return response.json()
    
class OpenAIAPI():
    def __init__(self, endpoint:str, token:str):
        self.endpoint = endpoint
        self.token = token

        self.client = OpenAI(api_key = token)

    def generate(self, model:str, prompt:str):
        response = self.client.responses.create(
                    model=model,
                    input=prompt
                )

        return response.output_text