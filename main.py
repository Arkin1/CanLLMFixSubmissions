
import pandas as pd
import datasets
import base64
from genai.utils import read_prompt
from genai.endpoints import get_llm_api
import hashlib
import sys
import json

from utils import ProblemsManager
import os
import dotenv
import asyncio
import logging
import yaml
import genai.methods as genai_methods
from genai.methods import LLMMethod
from tqdm import tqdm
import mlflow
import dspy
from sklearn.model_selection import train_test_split
from utils import preprocess_line

from data_models import Problem, Submission, GeneratedLLMResult, ResultAnalysis

logger = logging.getLogger(__name__)

def clean_source_code(source_code):
    num_lines_anchor = [preprocess_line(l) for l in source_code.splitlines()]
    num_lines_anchor = [l for l in num_lines_anchor if l != '']

    return "\n".join(num_lines_anchor)

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

    dataset_submissions['sourceCode'] = dataset_submissions['sourceCode'].apply(clean_source_code)

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

    manager = ProblemsManager(problems, generated_tests_ds)

    filter_subs = [manager.is_problem_usable(s['problem_id']) for _, s in dataset_submissions.iterrows()]
    dataset_submissions = dataset_submissions[filter_subs]

    return dataset_submissions, ProblemsManager(problems, generated_tests_ds)

async def predict(config):
    output_path = config['output']['path']
    os.makedirs(output_path, exist_ok=True)

    submissions, problem_manager = load_dataset(**config['dataset'])
    top_k = config['settings']['top_k']
    variance_per_sample = config['settings']['variance_per_sample']
    methods:list[LLMMethod] = [genai_methods.create_method(m_name, problems_manager = problem_manager, **m_args) for m_name, m_args in config['methods'].items()]
    filtered_submissions = submissions[(submissions['AcceptedAnchor'] != -1) & (submissions['verdict'] == 'WRONG_ANSWER')]

    problem_ids = filtered_submissions['problem_id']
    problem_ids = list(set(problem_ids))

    train_problem_ids, val_problem_ids = train_test_split(problem_ids, test_size = 0.75, random_state = 42)
    train_problem_ids = set(train_problem_ids)
    val_problem_ids = set(val_problem_ids)

    submissions_train = filtered_submissions[filtered_submissions['problem_id'].isin(train_problem_ids)]
    submissions_val = filtered_submissions[filtered_submissions['problem_id'].isin(val_problem_ids)]


    for idx, s in tqdm(list(submissions_val.iterrows())[:top_k]):
        try:
            problem = problem_manager.get_info_problem(s['problem_id'])
        except Exception as e:
            logger.error(f"Can't get problem {s['problem_id']}: {e}")
            continue

        s_idx = s['AcceptedAnchor']
        s_a = submissions.loc[s_idx]

        submission_anchor = Submission(submission_id = str(s_idx), 
                                        problem_id = problem.problem_id,
                                        source_code = s_a['sourceCode'], 
                                        programming_language = s_a['programmingLanguage'], 
                                        verdict = s_a['verdict'],
                                        ds_verdict = await problem_manager.evaluate_submission(s_a['problem_id'], s_a['sourceCode'], s_a['programmingLanguage'], 'hard'))
        
        submission = Submission(submission_id = str(idx), 
                                problem_id = problem.problem_id,
                                source_code = s['sourceCode'], 
                                programming_language = s['programmingLanguage'], 
                                verdict = s['verdict'],
                                ds_verdict = await problem_manager.evaluate_submission(s['problem_id'], s['sourceCode'], s['programmingLanguage'], 'hard'),
                                anchor = submission_anchor)
        
        if submission.ds_verdict == True:
            logger.warning(f'Submission with id {submission.submission_id} should not pass the tests')

        if submission.anchor.ds_verdict == False:
            logger.warning(f'Submission anchor with id {submission.anchor.submission_id} should pass the tests')

        for idx_variance in range(variance_per_sample):
            results  = []
            for method in methods:
                try:
                    llm_result = method.predict(submission)
                    llm_result.loss = await method.compute_loss(submission.problem_id, submission.source_code, submission.anchor.source_code, llm_result.source_code)
                    results.append(llm_result)
                except Exception as e:
                    logger.error(f'Could not predict submission {submission.submission_id} for problem {problem.problem_id}. Reason: {e}. Skipping!')
                    raise e

            analysis_result = ResultAnalysis(problem_id = problem.problem_id, submission=submission, generated_results=results)
            problem_output_path = os.path.join(output_path, problem.problem_id.replace('/', '_'))
            os.makedirs(problem_output_path, exist_ok=True)

            with open(os.path.join(problem_output_path, submission.submission_id + f'_{idx_variance}.json'), 'w', encoding='utf-8') as fp:
                fp.write( analysis_result.model_dump_json(indent = 1))


def evaluate(config):
    output_path = config['output']['path']

    buckets_samples_analyses = {}
    for root, _, files in os.walk(output_path):
        for f in files:
            with open(os.path.join(root, f), 'r', encoding='utf-8') as fp:
                result_analysis = ResultAnalysis.model_validate_json(fp.read())
                sub_id = result_analysis.submission.submission_id
                if sub_id not in buckets_samples_analyses:
                    buckets_samples_analyses[sub_id] = []
                buckets_samples_analyses[sub_id].append(result_analysis)
    
    result = []
    for sample_id, var_samples in buckets_samples_analyses.items():
        result_sample = {"sample_id": sample_id,
                         "problem_id": var_samples[0].problem_id}
        metrics = {}
        for s_result in var_samples:
            for generated_result in s_result.generated_results:
                if generated_result.method_name not in metrics:
                    metrics[generated_result.method_name] = []

                metrics[generated_result.method_name].append(generated_result.loss.model_dump())
        
        for m_name, values in metrics.items():
            df = pd.DataFrame.from_records(values)
            stats = df.describe()

            for col in stats.columns:
                for descriptor in stats.index:
                        result_sample[m_name + '_' + col + '_' + descriptor] = stats.loc[descriptor, col].item()

        result.append(result_sample)
    
    result_df = pd.DataFrame.from_records(result)
    result_df.to_csv(os.path.join(output_path, 'result.csv'), index=False)

async def fit(config):
    # Enable full autologging
    mlflow.dspy.autolog(
        log_compiles=True,           # Track the optimization process
        log_evals=True,              # Track evaluation results
        log_traces_from_compile=True # Trace every LLM call during optimization
    )

    mlflow.set_experiment(config["experiment_name"])

    output_path = config['output']['path']
    os.makedirs(output_path, exist_ok=True)

    submissions_df, problem_manager = load_dataset(**config['dataset'])

    filtered_submissions = submissions_df[(submissions_df['AcceptedAnchor'] != -1) & (submissions_df['verdict'] == 'WRONG_ANSWER')]

    filtered_submissions['ratio_mod'] = (filtered_submissions['code_deletions'] + filtered_submissions['code_additions']) / (2 * filtered_submissions['number_lines_anchor'])
    filtered_submissions = filtered_submissions[filtered_submissions['ratio_mod'] < 0.1]

    top_k = config['settings']['top_k']

    submissions = []
    for idx, s in tqdm(list(filtered_submissions.iterrows())[:top_k]):
        try:
            problem = problem_manager.get_info_problem(s['problem_id'])
        except Exception as e:
            logger.error(f"Can't get problem {s['problem_id']}: {e}")
            continue

        s_idx = s['AcceptedAnchor']
        s_a = submissions_df.loc[s_idx]

        submission_anchor = Submission(submission_id = str(s_idx),
                                       problem_id = problem.problem_id, 
                                        source_code = s_a['sourceCode'], 
                                        programming_language = s_a['programmingLanguage'], 
                                        verdict = s_a['verdict'],
                                        ds_verdict = True)
        
        submission = Submission(submission_id = str(idx), 
                                problem_id = problem.problem_id,
                                source_code = s['sourceCode'], 
                                programming_language = s['programmingLanguage'], 
                                verdict = s['verdict'],
                                ds_verdict = False,
                                anchor = submission_anchor)
        
        if submission.ds_verdict == True:
            logger.warning(f'Submission with id {submission.submission_id} should not pass the tests')

        if submission.anchor.ds_verdict == False:
            logger.warning(f'Submission anchor with id {submission.anchor.submission_id} should pass the tests')

        submissions.append(submission)
    
    model:LLMMethod = [genai_methods.create_method(m_name, problems_manager = problem_manager, **m_args) for m_name, m_args in config['fit'].items()][0]
    
    problem_ids = [s.problem_id for s in submissions]
    problem_ids = list(set(problem_ids))

    train_problem_ids, val_problem_ids = train_test_split(problem_ids, test_size = 0.75, random_state = 42)
    train_problem_ids = set(train_problem_ids)
    val_problem_ids = set(val_problem_ids)

    submissions_train = [s for s in submissions if s.problem_id in train_problem_ids]
    submissions_val = [s for s in submissions if s.problem_id in train_problem_ids]

    model.fit(submissions_train, submissions_val)
    

async def main():
    dotenv.load_dotenv(dotenv_path='.devcontainer/.env', override=True)
    with open('parameters.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    await predict(config)
    #evaluate(config)
    #await fit(config)

if __name__ == "__main__":
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    asyncio.run(main())
    
    



