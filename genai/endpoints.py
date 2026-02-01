
import os
import dspy


def _get_dspy_llm_openai(model_name:str, **kwargs):
    endpoint = os.environ.get('OPENAI_ENDPOINT')
    token = os.environ.get('OPENAI_TOKEN')

    return dspy.LM(f'openai/{model_name}', api_key = token, **kwargs)

def _get_dspy_llm_ollama(model_name:str, **kwargs):
    endpoint = os.environ.get('OLLAMA_ENDPOINT')
    token = os.environ.get('OLLAMA_TOKEN')

    return dspy.LM(f'ollama_chat/{model_name}', api_base = endpoint, api_key = token, **kwargs)

vendors = {
    "OpenAI": _get_dspy_llm_openai,
    "Ollama": _get_dspy_llm_ollama
}
    
def get_llm_api(vendor:str, model_name:str, **kwargs) -> dspy.BaseLM:
    return vendors[vendor](model_name, **kwargs)
