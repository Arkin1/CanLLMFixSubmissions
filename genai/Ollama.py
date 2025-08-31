
import requests

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