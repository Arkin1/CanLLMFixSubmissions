
import pandas as pd
import base64
import sys
from utils import ProblemsManager, attach_dif_data, df_to_submissions
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
from utils import get_problems_manager
from create_dataset_submissions import create_dataset
import argparse
from utils import clean_source_code
import seaborn as sns
import matplotlib.pyplot as plt

from data_models import Problem, Submission, GeneratedLLMResult, ResultAnalysis

logger = logging.getLogger(__name__)

def preprocess_data(dataset_submissions:pd.DataFrame, code_mod_ratio_threshold:float):
    dataset_submissions = dataset_submissions[~((dataset_submissions['verdict'] != 'OK') & (dataset_submissions['AcceptedAnchor']==-1))]

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
    
    dataset_submissions['sourceCodeOriginal'] = dataset_submissions['sourceCode']
    dataset_submissions['sourceCode'] = dataset_submissions['sourceCode'].apply(lambda x: base64.b64decode(x).decode('utf-8'))
    dataset_submissions['sourceCode'] = dataset_submissions['sourceCode'].apply(clean_source_code)
    attach_dif_data(dataset_submissions)
    dataset_submissions['sourceCode'] = dataset_submissions['sourceCode'].apply(lambda x: base64.b64encode(x.encode('utf-8')))

    dataset_submissions = dataset_submissions[dataset_submissions['code_similar_score'] >= code_mod_ratio_threshold]

    anchors_ids = set(dataset_submissions[dataset_submissions['AcceptedAnchor'] != -1]['AcceptedAnchor'])

    dataset_submissions = dataset_submissions[(dataset_submissions['verdict'] != 'OK') | 
                                              ((dataset_submissions['verdict'] == 'OK') & (dataset_submissions.index.isin(anchors_ids)))]

    return dataset_submissions

def load_dataset(path_to_dataset):
    dataset_submissions = pd.read_csv(path_to_dataset)
    dataset_submissions = dataset_submissions.set_index('submissions_id')

    return dataset_submissions

async def create_dataset_step(config):
    dataset_config = config['create_dataset']
    await create_dataset(dataset_config['path_to_submissions'], 
                         dataset_config['path_to_test_data'], 
                         dataset_config['path_to_contest_data'],
                         output_path=dataset_config['output_path'])

def compute_figures(df_sub:pd.DataFrame, output_path:str, suffix:str):
    print("Computing figures...")
    df_buggy = df_sub[df_sub["AcceptedAnchor"] != -1]
    plt.figure(figsize=(18, 12))

    g = sns.barplot(y = df_sub["verdict"].value_counts().index, x = df_sub["verdict"].value_counts().values, orient = 'y')
    plt.savefig(os.path.join(output_path, f'verdict_distribution_{suffix}.png'))
    plt.clf()
    
    t = df_buggy.groupby('AcceptedAnchor').size().reset_index(name='Number of buggy submissions per anchor')
    g = sns.histplot(t['Number of buggy submissions per anchor'])
    plt.savefig(os.path.join(output_path, f'buggy_submissions_per_anchor_{suffix}.png'))
    plt.clf()

    t = df_buggy.groupby(['AcceptedAnchor', 'verdict']).size().reset_index(name='Number of buggy submissions per anchor')
    g = sns.displot(t, x = 'Number of buggy submissions per anchor', hue = 'verdict', col = 'verdict',stat = 'density', common_norm=False)
    plt.savefig(os.path.join(output_path, f'buggy_submissions_per_anchor_{suffix}.png'))
    plt.clf()

    if "code_addition_ratio" in df_buggy.columns and "code_deletion_ratio" in df_buggy.columns and "code_similar_score" in df_buggy.columns:
        sns.displot(df_buggy, x = 'code_addition_ratio',hue = 'verdict', col = 'verdict', stat='density',common_norm=False)
        plt.savefig(os.path.join(output_path, f'code_addition_ratio_{suffix}.png'))
        plt.clf()

        sns.displot(df_buggy, x = 'code_deletion_ratio',hue = 'verdict', col = 'verdict', stat='density',common_norm=False)
        plt.savefig(os.path.join(output_path, f'code_deletion_ratio_{suffix}.png'))
        plt.clf()

        sns.displot(df_buggy, x = 'code_similar_score',hue = 'verdict', col = 'verdict', stat='density',common_norm=False)
        plt.savefig(os.path.join(output_path, f'code_similar_score_{suffix}.png'))
        plt.clf()

def preprocess_dataset_step(config):
    print("Preprocessing dataset...")
    config = config['preprocess_dataset']

    os.makedirs(config['output_figures_path'], exist_ok=True)

    dataset_submissions = load_dataset(config['path_to_dataset'])
    compute_figures(dataset_submissions, config['output_figures_path'], suffix = 'original')
    
    dataset_submissions = preprocess_data(dataset_submissions, config['code_mod_ratio_threshold'])
    compute_figures(dataset_submissions, config['output_figures_path'], suffix = 'preprocessed')

    problem_ids = dataset_submissions['problem_id'].unique()

    train_ratio = config['train_ratio']
    val_ratio = config['val_ratio']
    test_ratio = config['test_ratio']
    seed = config['seed']

    train_problems, test_problems = train_test_split(problem_ids, test_size=test_ratio, random_state=seed)
    train_problems, val_problems = train_test_split(train_problems, test_size=val_ratio/(train_ratio + val_ratio), random_state=seed)

    dataset_submissions_train = dataset_submissions[dataset_submissions['problem_id'].isin(train_problems)]
    dataset_submissions_val = dataset_submissions[dataset_submissions['problem_id'].isin(val_problems)]
    dataset_submissions_test = dataset_submissions[dataset_submissions['problem_id'].isin(test_problems)]

    dataset_submissions_train.to_csv(config['output_train_path'])
    compute_figures(dataset_submissions_train, config['output_figures_path'], suffix = 'train')

    dataset_submissions_val.to_csv(config['output_val_path'])
    compute_figures(dataset_submissions_val, config['output_figures_path'], suffix = 'val')

    dataset_submissions_test.to_csv(config['output_test_path'])
    compute_figures(dataset_submissions_test, config['output_figures_path'], suffix = 'test')
    
async def predict_step(config):
    output_path = config['output']['path']
    top_k = config['settings'].get('top_k')
    variance_per_sample = config['settings']['variance_per_sample']

    os.makedirs(output_path, exist_ok=True)

    r1_dataset_config = config['r1_codeforces_dataset']

    predict_config = config['predict']
    dataset_config = predict_config['dataset']

    submissions_df = load_dataset(dataset_config['path_to_submissions'])
    problems_ids = set(list(submissions_df['problem_id']))

    problem_manager = get_problems_manager(problems_ids, 
                                  path_to_contest_data=r1_dataset_config['path_to_contest_data'], 
                                  path_to_test_data=r1_dataset_config['path_to_test_data'],
                                  cache=r1_dataset_config['cache'],
                                  cache_folder=r1_dataset_config['cache_folder'])

    methods:list[LLMMethod] = [genai_methods.create_method(m_name, problems_manager = problem_manager, **m_args) for m_name, m_args in predict_config['methods'].items()]
    
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
                    llm_result.reward = await method.compute_reward(submission.problem_id, submission.source_code, submission.anchor.source_code, llm_result.source_code)
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

                metrics[generated_result.method_name].append(generated_result.reward.model_dump())
        
        for m_name, values in metrics.items():
            df = pd.DataFrame.from_records(values)
            stats = df.describe()

            for col in stats.columns:
                for descriptor in stats.index:
                        if descriptor == 'mean' or descriptor == 'std' or descriptor == 'min' or descriptor=='max':
                            if col == 'total_reward':
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
    top_k = config['settings'].get('top_k')
    os.makedirs(output_path, exist_ok=True)

    r1_dataset_config = config['r1_codeforces_dataset']


    fit_config = config['fit']
    train_dataset_config = fit_config['train_dataset']
    val_dataset_config = fit_config['val_dataset']

    submissions_df_train = load_dataset(train_dataset_config['path_to_submissions'])
    submissions_df_val = load_dataset(val_dataset_config['path_to_submissions'])

    problems_ids = set(list(submissions_df_train['problem_id']) + list(submissions_df_val['problem_id']))


    problem_manager = get_problems_manager(problems_ids, 
                                  path_to_contest_data=r1_dataset_config['path_to_contest_data'], 
                                  path_to_test_data=r1_dataset_config['path_to_test_data'],
                                  cache=r1_dataset_config['cache'],
                                  cache_folder=r1_dataset_config['cache_folder'])
        
    if top_k:
        submissions_train = df_to_submissions(submissions_df_train, problem_manager)[:top_k]
        submissions_val = df_to_submissions(submissions_df_val, problem_manager)[:top_k]
    else:
        submissions_train = df_to_submissions(submissions_df_train, problem_manager) 
        submissions_val = df_to_submissions(submissions_df_val, problem_manager) 

    model:LLMMethod = [genai_methods.create_method(m_name, problems_manager = problem_manager, **m_args) for m_name, m_args in fit_config['methods'].items()][0]
    
    model.fit(submissions_train, submissions_val)
    

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--create-dataset', action="store_true")
    parser.add_argument('--preprocess-dataset', action="store_true")
    parser.add_argument('--predict', action = "store_true")
    parser.add_argument('--evaluate', action = 'store_true')
    parser.add_argument('--fit', action = 'store_true')

    dotenv.load_dotenv(dotenv_path='.devcontainer/.env', override=True)
    with open('parameters.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    args = parser.parse_args(["--preprocess-dataset"])

    if args.create_dataset:
        await create_dataset_step(config)
    if args.preprocess_dataset:
        preprocess_dataset_step(config)
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
    
    



