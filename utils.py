import aiohttp
import asyncio
from typing import Optional, Dict
import difflib
import traceback
import datasets
import logging
from data_models import Problem

logger = logging.getLogger()

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

def count_differences(source_a:str, source_b:str):
    def preprocess(s:str):
        return s.replace('\t', '')

    lines_a = preprocess(source_a).splitlines()
    lines_b = preprocess(source_b).splitlines()

    lines_a = [l for l in lines_a if l != '']
    lines_b = [l for l in lines_b if l != '']

    diff_lines = difflib.unified_diff(lines_a, lines_b, "Original", "Modified")

    additions = 0
    deletions = 0
    
    # Parse diff lines to count additions and deletions
    for line in diff_lines:
        if line.startswith('+') and not line.startswith('+++'):
            additions += 1
        elif line.startswith('-') and not line.startswith('---'):
            deletions += 1

    return additions, deletions

class RetryContext():
    def __init__(self, max_number_times, logger = None):
        self.max_number_times = max_number_times
        self.logger = logger
        self.num_retries = 0
        self.errors = []

    def __call__(self, f):
        while self.num_retries <= self.max_number_times:
            try:
                return f()
            except:
                err = traceback.format_exc()
                if self.logger:
                    self.logger.warning(f"Retry {self.num_retries} / {self.max_number_times} with error: {err}")
                self.errors.append(err)
                self.num_retries+=1
                continue
        return None
    
class RetryContextAsync():
    def __init__(self, max_number_times, logger = None):
        self.max_number_times = max_number_times
        self.logger = logger
        self.num_retries = 0
        self.errors = []

    async def __call__(self, f):
        while self.num_retries <= self.max_number_times:
            try:
                return await f()
            except:
                err = traceback.format_exc()
                if self.logger:
                    self.logger.warning(f"Retry {self.num_retries} / {self.max_number_times} with error: {err}")
                self.errors.append(err)
                self.num_retries+=1
                continue
        return None

class RetryException(Exception):
    pass

class RetryExecution():
    def __init__(self, context_name, max_number_times, logger = None):
        self.context_name = context_name
        self.max_number_times = max_number_times
        self.logger = logger
    
    def __enter__(self):
        self.context = RetryContext(self.max_number_times, self.logger)
        return self.context

    def __exit__(self, exc_type, exc_value, traceback):
        if self.context.num_retries > self.max_number_times:
            if self.logger:
                for idx, e in enumerate(self.context.errors):
                    self.logger.error(f"Retry {idx} / {self.context.max_number_times} with error: {e}")
                self.logger.error(f"Number of retries exceeded for context {self.context_name}")
            raise RetryException(f"Number of retries exceeded for context {self.context_name}!")
        
        return False
    
class RetryExecutionAsync():
    def __init__(self, context_name, max_number_times, logger = None):
        self.context_name = context_name
        self.max_number_times = max_number_times
        self.logger = logger
    
    def __enter__(self):
        self.context = RetryContextAsync(self.max_number_times, self.logger)
        return self.context

    def __exit__(self, exc_type, exc_value, traceback):
        if self.context.num_retries > self.max_number_times:
            if self.logger:
                for idx, e in enumerate(self.context.errors):
                    self.logger.error(f"Retry {idx} / {self.context.max_number_times} with error: {e}")
                self.logger.error(f"Number of retries exceeded for context {self.context_name}")
            raise RetryException(f"Number of retries exceeded for context {self.context_name}!")
        
        return False
    
class ProblemsManager():
    def __init__(self, 
                 problems,
                 generated_tests_ds,
                 endpoint_eval = "http://host.docker.internal:2000",
                 ):
        self.problems = problems
        self.generated_tests_ds = generated_tests_ds['test']
        
        self.problems_train = self.problems['train']
        self.problems_test = self.problems['test']
        self.endpoint_eval = endpoint_eval

        self._create_indices()

    def get_info_problem(self, problem_id:str) -> Problem:
        if problem_id in self.problems_train_ds_index_mapping:
            problem_info = self.problems_train[self.problems_train_ds_index_mapping[problem_id][0]]
        else:
            problem_info = self.problems_train[self.problems_train_ds_index_mapping[problem_id][0]]

        tests = problem_info['official_tests']

        if not problem_info['official_tests_complete']:
            if problem_info['generated_tests'] > 0:
                for idx in self.generated_tests_ds_index_mapping[problem_id]:
                    test_data = self.generated_tests_ds[idx]
                    tests.append({"input": test_data['input'], "output": test_data['output']})

        return Problem(problem_info['id'],
                       problem_info['description'],
                       problem_info['input_format'],
                       problem_info['output_format'],
                       problem_info['examples'],
                       problem_info['note'],
                       tests)

    async def evaluate_submission(self, problem_id:str, source_code:str, programming_language:str):
        problem_info, tests = self.get_info_problem(problem_id)

        with RetryExecutionAsync("problem_evaluator", 3, logger) as retry_async:
            async def eval():
                response_anchor = await compile_and_test(source_code, problem_info, tests, self.endpoint_eval, programming_language)
                r_anchor = self._check(response_anchor)
                return r_anchor
            
            return await retry_async(eval)

    def _check(self, response):
        return all([result and 'compile' in result and result['compile']['code'] == 0 and result['run']['code'] == 0 and result['run']['stdout'].split()[0] == '1' for result in response])

    def _create_indices(self):
        self.generated_tests_ds_index_mapping = self._create_index(self.generated_tests_ds, 'problem_id')
        self.problems_train_ds_index_mapping = self._create_index(self.problems_train, 'id')
        self.problems_test_ds_index_mapping = self._create_index(self.problems_test, 'id')

    def _create_index(self, ds, index_name):
        index_c = {}

        for idx, problem_id in enumerate(ds[index_name]):
            if problem_id not in index_c:
                index_c[problem_id] = []

            index_c[problem_id].append(idx)
        
        return index_c
