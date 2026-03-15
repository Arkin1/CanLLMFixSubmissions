import re

class CodeParser():
    re_code1 = re.compile(r'```c\+\+(.+)```')
    re_code2 = re.compile(r'```cpp(.+)```')
    re_code3 = re.compile(r'```c(.+)```')
    re_code4 = re.compile(r'```(.+)```')
    think_code4 = re.compile(r'```<think>(.+)</think>```')

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

        if len(found_code) == 0:
            return s.replace('\\\\n', '\n')
        else:
            return found_code[0].replace('\\\\n', '\n')