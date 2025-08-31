import grequests
import pandas as pd
import datasets
from utils import compile_and_test
import base64

from genai.utils import read_prompt
from genai.Ollama import OllamaAPI

def create_index(ds, index_name):
    index_c = {}

    for idx, problem_id in enumerate(ds[index_name]):
        if problem_id not in index_c:
            index_c[problem_id] = []

        index_c[problem_id].append(idx)
    
    return index_c


dataset_submissions = pd.read_csv('dataset.csv')
dataset_submissions = dataset_submissions.set_index('submissions_id')
dataset_submissions['total_code_mod'] = dataset_submissions['code_additions'] + dataset_submissions['code_deletions']

dataset_submissions = dataset_submissions.sort_values('total_code_mod')

generated_tests_ds = datasets.load_dataset("data/codeforces/generated_tests")['test']

generated_tests_ds_index_mapping = create_index(generated_tests_ds, 'problem_id')

all_contests = set([k.split('/')[0] for k in generated_tests_ds_index_mapping.keys()])

problems_train = datasets.load_dataset("data/codeforces/data")['train']
problems_test = datasets.load_dataset("data/codeforces/data")['test']

problems_train_ds_index_mapping = create_index(problems_train, 'id')
problems_test_ds_index_mapping = create_index(problems_test, 'id')

ollama_api = OllamaAPI('','')

for _, s in dataset_submissions.iterrows():
    if s['AcceptedAnchor'] == -1 or s['verdict'] !='WRONG_ANSWER':
        continue

    if s['problem_id'] in problems_train_ds_index_mapping:
        problem_data = problems_train[problems_train_ds_index_mapping[s['problem_id']][0]]
    elif s['problem_id'] in problems_test_ds_index_mapping:
        problem_data = problems_test[problems_test_ds_index_mapping[s['problem_id']][0]]
    else:
        continue
    
    source_code = base64.b64decode(s['sourceCode']).decode('utf-8')
    anchor_source_code = base64.b64decode(dataset_submissions.loc[s['AcceptedAnchor']]['sourceCode']).decode('utf-8')

    oficial_tests = problem_data['official_tests']

    if not problem_data['official_tests_complete']:
        if problem_data['generated_tests'] > 0:
            for idx in generated_tests_ds_index_mapping[s['problem_id']]:
                test_data = generated_tests_ds[idx]
                oficial_tests.append({"input": test_data['input'], "output": test_data['output']})
    
    def check(response):
        return all([result and 'compile' in result and result['compile']['code'] == 0 and result['run']['code'] == 0 and result['run']['stdout'].split()[0] == '1' for result in response])

    response_anchor = compile_and_test(anchor_source_code, problem_data, oficial_tests, "http://host.docker.internal:2000")
    r_anchor = check(response_anchor)
    
    response_wrong = compile_and_test(source_code, problem_data, oficial_tests, "http://host.docker.internal:2000")
    r_wrong = check(response_wrong)

    prompt = read_prompt('prompt_naive_fix_bug',
                         problem_description = problem_data['description'],
                         problem_input_format = problem_data['input_format'],
                         problem_output_format = problem_data['output_format'],
                         problem_example = problem_data['examples'],
                         note = problem_data['note'],
                         submission_verdict = s['verdict'],
                         submission_code = source_code)
    
    response_generaged = ollama_api.generate('llama3.2:3b', prompt)['response']
    

    
    print(f"original ({s['verdict']}) -> judge anchor ({'OK' if r_anchor == True else 'wrong/TLE/MLE'}) -> judge wrong ({'OK' if r_wrong== True else 'wrong/TLE/MLE'})")
        





