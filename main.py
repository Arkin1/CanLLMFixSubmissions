
import pandas as pd
import base64
import sys
from utils import ProblemsManager, attach_dif_counts
import os
import dotenv
import asyncio
import logging
import yaml
import genai.methods as genai_methods
from genai.methods import LLMMethod
from tqdm import tqdm
import mlflow
from sklearn.model_selection import train_test_split
from utils import  get_codeforces_r1_dataset
from create_dataset_submissions import create_dataset
import argparse
from utils import clean_source_code

from data_models import Problem, Submission, GeneratedLLMResult, ResultAnalysis

logger = logging.getLogger(__name__)

def load_dataset(path_to_dataset, 
                 path_to_test_data, 
                 path_to_contest_data, 
                 cache_folder = 'data/cache',
                 cache = True):
    dataset_submissions = pd.read_csv(path_to_dataset)
    dataset_submissions = dataset_submissions.set_index('submissions_id')

    dataset_submissions['sourceCode'] = dataset_submissions['sourceCode'].apply(lambda x: base64.b64decode(x).decode('utf-8'))
    dataset_submissions['sourceCode'] = dataset_submissions['sourceCode'].apply(clean_source_code)

    problems_ids = set(list(dataset_submissions['problem_id']))

    manager = get_codeforces_r1_dataset(problems_ids, path_to_contest_data, path_to_test_data)

    dataset_submissions = dataset_submissions[~((dataset_submissions['verdict'] != 'OK') & dataset_submissions['AcceptedAnchor']==-1)]

    dataset_submissions = dataset_submissions[(dataset_submissions['is_problem_usable'] & 
                                               (((dataset_submissions['verdict'] == "OK") & (dataset_submissions['passes_r1_tests'] == True)) | 
                                                (((dataset_submissions['verdict'] != "OK") & (dataset_submissions['passes_r1_tests'] == False)))))
                                              ]
    dataset_submissions['has_anchor'] = dataset_submissions.apply(lambda f: (f['AcceptedAnchor'] == -1 or 
                                                                            (f['AcceptedAnchor'] != -1 and f['AcceptedAnchor'] in dataset_submissions.index)), axis = 1)
    dataset_submissions = dataset_submissions[dataset_submissions['has_anchor']]
    dataset_submissions = dataset_submissions.drop('has_anchor', axis = 1)
    dataset_submissions = dataset_submissions.loc[~dataset_submissions.index.duplicated(keep='first'), :]

    dataset_submissions = dataset_submissions[dataset_submissions["verdict"].isin(["OK", 
                                        "WRONG_ANSWER", 
                                        "TIME_LIMIT_EXCEEDED", 
                                        "RUNTIME_ERROR",
                                        "MEMORY_LIMIT_EXCEEDED",
                                        "COMPILATION_ERROR"])]
    
    attach_dif_counts(dataset_submissions)
    
    return dataset_submissions, manager

def df_to_submissions(submissions_df:pd.DataFrame, problem_manager:ProblemsManager) -> list[Submission]:
    submissions = []
    for idx, s in tqdm(list(submissions_df.iterrows())):
        try:
            problem = problem_manager.get_info_problem(s['problem_id'])
        except Exception as e:
            logger.error(f"Can't get problem {s['problem_id']}: {e}")
            continue

        s_idx = s['AcceptedAnchor']
        if s_idx == -1:
            continue
        s_a = submissions_df.loc[s_idx]

        submission_anchor = Submission(submission_id = str(s_idx), 
                                        problem_id = problem.problem_id,
                                        source_code = s_a['sourceCode'], 
                                        programming_language = s_a['programmingLanguage'], 
                                        verdict = s_a['verdict'],
                                        ds_verdict = s_a['passes_r1_tests'])
        
        submission = Submission(submission_id = str(idx), 
                                problem_id = problem.problem_id,
                                source_code = s['sourceCode'], 
                                programming_language = s['programmingLanguage'], 
                                verdict = s['verdict'],
                                ds_verdict = s['passes_r1_tests'],
                                anchor = submission_anchor)
        
        if submission.ds_verdict == True:
            logger.warning(f'Submission with id {submission.submission_id} should not pass the tests')

        if submission.anchor.ds_verdict == False:
            logger.warning(f'Submission anchor with id {submission.anchor.submission_id} should pass the tests')
    
        submissions.append(submission)
    return submissions

async def create_dataset_step(config):
    dataset_config = config['create_dataset']
    await create_dataset(dataset_config['path_to_submissions'], dataset_config['path_to_test_data'], dataset_config['path_to_contest_data'])

async def predict_step(config):
    output_path = config['output']['path']
    os.makedirs(output_path, exist_ok=True)

    submissions_df, problem_manager = load_dataset(**config['dataset'])
    top_k = config['settings'].get('top_k')
    variance_per_sample = config['settings']['variance_per_sample']
    methods:list[LLMMethod] = [genai_methods.create_method(m_name, problems_manager = problem_manager, **m_args) for m_name, m_args in config['methods'].items()]
    
    print("Preprocessing data...")
    if top_k:
        submissions = df_to_submissions(submissions_df, problem_manager)
        submissions = submissions[:top_k]
    else:
        submissions = df_to_submissions(submissions_df, problem_manager) 

    print("Predicting...")
    for submission in tqdm(submissions):
        for idx_variance in range(variance_per_sample):
            results  = []
            for method in methods:
                try:
                    llm_result = method.predict(submission)
                    llm_result.loss = await method.compute_loss(submission.problem_id, submission.source_code, submission.anchor.source_code, llm_result.source_code)
                    results.append(llm_result)
                except Exception as e:
                    logger.error(f'Could not predict submission {submission.submission_id} for problem {submission.problem_id}. Reason: {e}. Skipping!')
                    raise e

            analysis_result = ResultAnalysis(problem_id = submission.problem_id, submission=submission, generated_results=results)
            problem_output_path = os.path.join(output_path, submission.problem_id.replace('/', '_'))
            os.makedirs(problem_output_path, exist_ok=True)

            with open(os.path.join(problem_output_path, submission.submission_id + f'_{idx_variance}.json'), 'w', encoding='utf-8') as fp:
                fp.write( analysis_result.model_dump_json(indent = 1))


def evaluate_step(config):
    output_path = config['output']['path']

    buckets_samples_analyses = {}
    for root, _, files in os.walk(output_path):
        for f in files:
            if f.endswith('.json'):
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
                        if descriptor == 'mean' or descriptor == 'std' or descriptor == 'min' or descriptor=='max':
                            if col == 'total_loss':
                                result_sample[m_name + '_' + col + '_' + descriptor] = stats.loc[descriptor, col].item()

        result.append(result_sample)
    
    result_df = pd.DataFrame.from_records(result)
    result_df.to_csv(os.path.join(output_path, 'result.csv'), index=False)

async def fit_step(config):
    # Enable full autologging
    mlflow.dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces_from_compile=True
    )

    mlflow.set_experiment(config["experiment_name"])

    output_path = config['output']['path']
    os.makedirs(output_path, exist_ok=True)

    top_k =config['settings'].get('top_k')

    submissions_df, problem_manager = load_dataset(**config['dataset'])

    #filtered_submissions = submissions_df[((submissions_df['AcceptedAnchor'] != -1) & (submissions_df['verdict'] == 'WRONG_ANSWER')) | (submissions_df['AcceptedAnchor'] == -1)]
    
    filtered_submissions  = submissions_df
    filtered_submissions['ratio_mod'] = (filtered_submissions['code_deletions'] + filtered_submissions['code_additions']) / (2 * filtered_submissions['number_lines_anchor'])
    filtered_submissions = filtered_submissions[(filtered_submissions['ratio_mod'] < 0.1) | (filtered_submissions['AcceptedAnchor'] == -1)]
    
    if top_k:
        submissions = df_to_submissions(filtered_submissions, problem_manager)[:top_k]
    else:
        submissions = df_to_submissions(filtered_submissions, problem_manager) 
    
    model:LLMMethod = [genai_methods.create_method(m_name, problems_manager = problem_manager, **m_args) for m_name, m_args in config['fit'].items()][0]
    
    problem_ids = [s.problem_id for s in submissions]
    problem_ids = list(set(problem_ids))

    train_problem_ids, val_problem_ids = train_test_split(problem_ids, test_size = 0.25, random_state = 42)
    train_problem_ids = set(train_problem_ids)
    val_problem_ids = set(val_problem_ids)

    submissions_train = [s for s in submissions if s.problem_id in train_problem_ids]
    submissions_val = [s for s in submissions if s.problem_id in train_problem_ids]

    model.fit(submissions_train, submissions_val)
    

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--create-dataset', action="store_true")
    parser.add_argument('--predict', action = "store_true")
    parser.add_argument('--evaluate', action = 'store_true')
    parser.add_argument('--fit', action = 'store_true')

    dotenv.load_dotenv(dotenv_path='.devcontainer/.env', override=True)
    with open('parameters.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    args = parser.parse_args(["--evaluate"])

    if args.create_dataset:
        await create_dataset_step(config)
    if args.predict:
        await predict_step(config)
    if args.evaluate:
        evaluate_step(config)
    if args.fit:
        await fit_step(config)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout
    )

    asyncio.run(main())
    
    



