
import requests
from openai import OpenAI
import os

class LLMApi():
    def __init__(self, model_name:str):
        self.model_name = model_name

    def generate(self, prompt:str):
        pass

class OllamaAPI(LLMApi):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.vendor = 'Ollama'

        self.endpoint = os.environ.get('OLLAMA_ENDPOINT')
        self.token = os.environ.get('OLLAMA_TOKEN')

    def generate(self, prompt:str):
        response = requests.post(self.endpoint, 
                                 headers={"Authorization": f"Bearer {self.token}"},
                                 json = {
                                     "model":self.model_name,
                                     "prompt": prompt,
                                     "stream": False
                                 })
        response.raise_for_status()

        return response.json()
    
class OpenAIAPI(LLMApi):
    def __init__(self, model_name:str):
        super().__init__(model_name)
        self.vendor = 'OpenAI'

        self.endpoint = os.environ.get('OPENAI_ENDPOINT')
        self.token = os.environ.get('OPENAI_TOKEN')

    def generate(self, prompt:str):

        client = OpenAI(api_key = self.token)

        response = client.responses.create(
                    model=self.model_name,
                    input=prompt
                )

        return response.output_text
    
vendors = {
    "OpenAI": OpenAIAPI,
    "Ollama": OllamaAPI
}
    
def get_llm_api(vendor:str, model_name:str) -> LLMApi:
    return vendors[vendor](model_name)