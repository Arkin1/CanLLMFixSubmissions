import aiohttp
import asyncio
from typing import Optional, Dict
import difflib
import traceback
import datasets
import logging
from data_models import Problem, Test
from typing import Literal
import hashlib
import os

logger = logging.getLogger()

async def _fetch(session, url, data, headers: Optional[Dict[str, str]] = None):
    async with session.post(url, json = data, headers =headers) as response:
        return await response.json()
    
async def _compile_and_test_async(source_code:str, problem_data:Problem, endpoint:str, programmingLanguage: str):
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
        logger.warning(f"Unsupported language {programmingLanguage}! Trying with cpp.")
        extension, piston_language = "cpp", "cf_c++"
    results = []
    async with aiohttp.ClientSession() as session:
        for test_case in problem_data.tests:
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
                        "content": test_case.input
                    },
                    {
                        "name": "correct_output.txt", 
                        "content": test_case.output
                    },
                    *([{"name": "checker.py", "content": problem_data.generated_checker}] if problem_data.generated_checker else []),
                    {
                        "name": "grader_config",
                        "content": "\n".join(
                            f"{key}={value}" for key, value in {
                                "TIME_LIMIT": problem_data.time_limit,
                                "MEMORY_LIMIT": problem_data.memory_limit,
                                "INPUT_MODE": problem_data.input_mode
                            }.items()
                        )
                    }
                ]
            }
            
            result = _fetch(session, f"{endpoint}/api/v2/execute", data=payload, headers={"Content-Type": "application/json"})
            results.append(result)

        results = await asyncio.gather(*results, return_exceptions=True)

        return results
    
async def compile_and_test(source_code:str, problem_data:Problem, endpoint:str, programmingLanguage:str):
    return await _compile_and_test_async(source_code, problem_data, endpoint, programmingLanguage)

def preprocess_line(s:str):
    s = s.replace('\t', '')
    s = s.replace('\r', '')
    s = s.strip()
    return s

def count_differences(source_a:str, source_b:str):
    diff_lines = difflib.unified_diff(source_a.splitlines(), source_b.splitlines(), "Original", "Modified")

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
    

def get_codeforces_r1_dataset(problems_ids:list[str], 
                              path_to_contest_data:str, 
                              path_to_test_files_data:str, 
                              cache:bool = True,
                              cache_folder = 'data/cache'):
    
    logger.info(f"Using cache {cache} and saving to {cache_folder}")
    
    cache_key = str(hashlib.sha1("-".join(sorted(list(problems_ids))).encode()).hexdigest())
    cache_path = os.path.join(cache_folder, cache_key)

    cache_problems_folder = os.path.join(cache_folder, cache_key, 'problems')
    cache_tests_folder = os.path.join(cache_folder, cache_key, 'tests')

    if cache and os.path.exists(cache_path):
        logger.info("Loading cache problems and test files...")
        problems = datasets.load_from_disk(cache_problems_folder)
        generated_tests_ds = datasets.load_from_disk(cache_tests_folder)

        return ProblemsManager(problems, generated_tests_ds)
    else:
        logger.info("Cache doesn't exist!")
        problems = datasets.load_dataset(path_to_contest_data)
        problems = problems.filter(lambda x: x['id'] in problems_ids)

        generated_tests_ds = datasets.load_dataset(path_to_test_files_data)
        generated_tests_ds = generated_tests_ds.filter(lambda x: x['problem_id'] in problems_ids)
        
        if cache:
            logger.info("Caching problems...")
            os.makedirs(cache_problems_folder)
            os.makedirs(cache_tests_folder)

            problems.save_to_disk(cache_problems_folder)
            generated_tests_ds.save_to_disk(cache_tests_folder)

            del problems
            del generated_tests_ds
            return get_codeforces_r1_dataset(problems_ids, path_to_contest_data, path_to_test_files_data, cache, cache_folder)
        else:
            logger.info("Caching is disabled")
            return ProblemsManager(problems, generated_tests_ds)


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
        elif problem_id in self.problems_test_ds_index_mapping:
            problem_info = self.problems_test[self.problems_test_ds_index_mapping[problem_id][0]]

        tests = [Test(input = t['input'], output = t['output']) for t in problem_info['official_tests']]

        if not problem_info['official_tests_complete']:
            if problem_info['generated_tests'] > 0:
                for idx in self.generated_tests_ds_index_mapping[problem_id]:
                    test_data = self.generated_tests_ds[idx]
                    tests.append(Test(input = test_data['input'], output = test_data['output']))

        return Problem(problem_id = problem_info['id'],
                       description = problem_info['description'],
                       input_format = problem_info['input_format'],
                       output_format = problem_info['output_format'],
                       examples = [Test(input = e['input'], output = e['output']) for e in problem_info['examples']],
                       note = problem_info['note'],
                       tests = tests,
                       time_limit=problem_info['time_limit'],
                       memory_limit=problem_info['memory_limit'],
                       input_mode=problem_info['input_mode'],
                       generated_checker=problem_info['generated_checker'])
    
    def is_problem_usable(self, problem_id:str) -> bool:
        if problem_id in self.problems_train_ds_index_mapping:
            problem_info = self.problems_train[self.problems_train_ds_index_mapping[problem_id][0]]
        elif problem_id in self.problems_test_ds_index_mapping:
            problem_info = self.problems_test[self.problems_test_ds_index_mapping[problem_id][0]]
        else:
            return False

        if (problem_info['description'] == None or 
           problem_info['input_format'] == None or
           problem_info['output_format'] == None or 
           problem_info['examples'] == None or
           problem_info['official_tests'] == None or 
           len(problem_info['official_tests']) == 0):
            return False

        return True
        
    async def evaluate_submission(self, problem_id:str, source_code:str, programming_language:str, evaluation_type: Literal['hard', 'soft']):
        problem_info = self.get_info_problem(problem_id)

        with RetryExecutionAsync("problem_evaluator", 3, logger) as retry_async:
            async def eval():
                response = await compile_and_test(source_code, problem_info, self.endpoint_eval, programming_language)
                if evaluation_type == 'hard':
                    score = self._hard_eval(response)
                else:
                    score = self._soft_eval(response)
                return score
            
            return await retry_async(eval)

    def _hard_eval(self, response):
        return all([result and 'compile' in result and result['compile']['code'] == 0 and result['run']['code'] == 0 and result['run']['stdout'].split()[0] == '1' for result in response])

    def _soft_eval(self, response):
        results = [result and 'compile' in result and result['compile']['code'] == 0 and result['run']['code'] == 0 and result['run']['stdout'].split()[0] == '1' for result in response]
        
        num_correct = 0

        for r in results:  
            if r:
                num_correct+=1

        return  num_correct / len(results)

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