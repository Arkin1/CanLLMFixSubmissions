
import pandas as pd
import datasets
import base64
from genai.utils import read_prompt
from genai.endpoints import get_llm_api
import hashlib
import sys

from utils import ProblemsManager
import os
import dotenv
import asyncio
import logging
import yaml
import genai.methods as genai_methods

from data_models import Problem, Submission, GeneratedLLMResult, ResultAnalysis

logger = logging.getLogger(__name__)

def load_dataset(path_to_submissions, 
                 path_to_test_data, 
                 path_to_contest_data, 
                 cache_folder = 'data/cache',
                 cache = True):
    dataset_submissions = pd.read_csv(path_to_submissions)
    dataset_submissions = dataset_submissions.set_index('submissions_id')
    dataset_submissions['total_code_mod'] = dataset_submissions['code_additions'] + dataset_submissions['code_deletions']
    dataset_submissions = dataset_submissions.sort_values('total_code_mod')

    dataset_submissions['sourceCode'] = dataset_submissions['sourceCode'].apply(lambda x: base64.b64decode(x).decode('utf-8'))

    problems_ids = set(list(dataset_submissions['problem_id']))

    cache_key = str(hashlib.sha1("-".join(sorted(list(problems_ids))).encode()).hexdigest())
    cache_path = os.path.join(cache_folder, cache_key)

    cache_problems_folder = os.path.join(cache_folder, cache_key, 'problems')
    cache_tests_folder = os.path.join(cache_folder, cache_key, 'tests')

    if cache and os.path.exists(cache_path):
        problems = datasets.load_from_disk(cache_problems_folder)
        generated_tests_ds = datasets.load_from_disk(cache_tests_folder)
    else:
        problems = datasets.load_dataset(path_to_contest_data)
        problems = problems.filter(lambda x: x['id'] in problems_ids)

        generated_tests_ds = datasets.load_dataset(path_to_test_data)
        generated_tests_ds = generated_tests_ds.filter(lambda x: x['problem_id'] in problems_ids)
        
        if cache:
            os.makedirs(cache_problems_folder)
            os.makedirs(cache_tests_folder)

            problems.save_to_disk(cache_problems_folder)
            generated_tests_ds.save_to_disk(cache_tests_folder)

            del problems
            del generated_tests_ds
            load_dataset(path_to_submissions, path_to_test_data, path_to_contest_data)

    return dataset_submissions, ProblemsManager(problems, generated_tests_ds)


async def main():
    dotenv.load_dotenv(dotenv_path='.devcontainer/.env', override=True)
    with open('parameters.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    submissions, problem_manager = load_dataset(**config['dataset'])

    methods = [genai_methods.create_method(m_name, **m_args) for m_name, m_args in config['methods'].items()]

    for idx, s in submissions.iterrows():
        problem = problem_manager.get_info_problem(s['problem_id'])
        submission = Submission(str(idx), s['sourceCode'], s['verdict'], None)
        
        results  = []
        for method in methods:
            try:
                llm_result = method.predict(problem, submission)
            except:
                logger.error(f'Could not predict submission {submission.submission_id} for problem {problem.problem_id}. Skipping!')
                results.append(None)
            results.append(llm_result)
        results = results



if __name__ == "__main__":
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    asyncio.run(main())
    
    



