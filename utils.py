import aiohttp
import asyncio
from typing import Optional, Dict

async def _fetch(session, url, data, headers: Optional[Dict[str, str]] = None):
    async with session.post(url, json = data, headers =headers) as response:
        return await response.json()


async def _compile_and_test_async(source_code:str, problem_data:dict, test_cases:list[dict], endpoint:str, programmingLanguage: str):
    if programmingLanguage == 'GNU C' or programmingLanguage == 'GNU C++' or programmingLanguage == 'MS C++':
        extension, piston_language = "cpp", "cf_c++"
    elif programmingLanguage == 'GNU C++0x':
        extension, piston_language = "cpp", "cf_c++0x"
    elif programmingLanguage == 'GNU C++11':
        extension, piston_language = "cpp", "cf_c++11"
    elif programmingLanguage == 'C++14 (GCC 6-32)':
        extension, piston_language = "cpp", "cf_c++14"
    elif programmingLanguage == 'C++17 (GCC 7-32)' or programmingLanguage == 'C++17 (GCC 9-64)' or programmingLanguage == 'Clang++17 Diagnostics' or programmingLanguage == 'GNU C++17 Diagnostics' or programmingLanguage == 'MS C++ 2017':
        extension, piston_language = "cpp", "cf_c++17"
    elif programmingLanguage == 'C++20 (GCC 11-64)' or programmingLanguage == 'C++20 (GCC 13-64)' or programmingLanguage == 'Clang++20 Diagnostics':
        extension, piston_language = "cpp", "cf_c++20"
    else:
        print(f"Unsupported language {programmingLanguage}!")
        raise Exception(f"Unsupported language {programmingLanguage}!")
    results = []
    async with aiohttp.ClientSession() as session:
        for test_case in test_cases:
            payload = {
                "language": piston_language,
                "version": "*", 
                "files": [
                    {
                        "name": f"main.{extension}",
                        "content": source_code
                    },
                    {
                        "name": "input.txt",
                        "content": test_case['input']
                    },
                    {
                        "name": "correct_output.txt", 
                        "content": test_case['output']
                    },
                    *([{"name": "checker.py", "content": problem_data['generated_checker']}] if problem_data['generated_checker'] else []),
                    {
                        "name": "grader_config",
                        "content": "\n".join(
                            f"{key}={value}" for key, value in {
                                "TIME_LIMIT": problem_data['time_limit'],
                                "MEMORY_LIMIT": problem_data['memory_limit'],
                                "INPUT_MODE": problem_data['input_mode']
                            }.items()
                        )
                    }
                ]
            }
            
            result = _fetch(session, f"{endpoint}/api/v2/execute", data=payload, headers={"Content-Type": "application/json"})
            results.append(result)

        results = await asyncio.gather(*results, return_exceptions=True)

        return results
    
async def compile_and_test(source_code:str, problem_data:dict, test_cases:list[dict], endpoint:str, programmingLanguage:str):
    return await _compile_and_test_async(source_code, problem_data, test_cases, endpoint, programmingLanguage)