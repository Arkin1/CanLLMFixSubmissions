
def read_prompt(name:str, **kwargs):
    with open(f'prompts/{name}.txt', 'r') as fp:
        prompt_txt = fp.read()
    
    for k,v in kwargs.items():
        prompt_txt = prompt_txt.replace(f'%%%{k.upper()}%%%', v)
    
    return prompt_txt