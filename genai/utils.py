import re

def read_prompt(name:str, **kwargs):
    with open(f'prompts/{name}.txt', 'r') as fp:
        prompt_txt = fp.read()
    
    for k,v in kwargs.items():
        prompt_txt = prompt_txt.replace(f'%%%{k.upper()}%%%', v)
    
    return prompt_txt


class CodeParser():
    re_code1 = re.compile(r'```c\+\+(.+)```')
    re_code2 = re.compile(r'```cpp(.+)```')
    re_code3 = re.compile(r'```c(.+)```')
    re_code4 = re.compile(r'```(.+)```')
    think_code4 = re.compile((r'```<think>(.+)</think>```'))

    @staticmethod
    def extract_code(s:str):
        s = s.replace('\n', '\\\\n')
        s = CodeParser.think_code4.sub('', s)
        found_code = CodeParser.re_code1.findall(s)
        if len(found_code) == 0:
            found_code = CodeParser.re_code2.findall(s)
        
        if len(found_code) == 0:
            found_code = CodeParser.re_code3.findall(s)
        
        if len(found_code) == 0:
            found_code = CodeParser.re_code4.findall(s)

        return found_code[0].replace('\\\\n', '\n')