import pandas as pd
import base64
import logging
from glob import glob
import json
import os
from utils import get_codeforces_r1_dataset
from tqdm import tqdm


logger = logging.getLogger()

def _determine_anchor(group):
    group_creation_dates = [(idx, e['verdict'], e['creationTimeSeconds']) for idx, e in group.iterrows()]
    anchors = [-1 for _, _ in group.iterrows()]

    group_creation_dates = sorted(group_creation_dates, key = lambda x: x[2], reverse=True)

    last_accepted_idx = -1

    for idx, (_, verdict, _) in enumerate(group_creation_dates):
        if verdict == 'OK':
            last_accepted_idx = idx
        elif last_accepted_idx != -1:
            anchors[idx] = last_accepted_idx

    anchors = [group.iloc[a]['submissions_id'] if a != -1 else -1 for a in anchors]

    group['AcceptedAnchor'] = anchors

    return group

async def _append_submission_status_r1(dataset:pd.DataFrame, path_to_test_files_data:str, path_to_contest_data:str):
    logger.info(f"Marking submissions that can be used with Codeforces R1 Dataset and if they pass the test suite or not...")
    problems_ids = set(list(dataset['problem_id']))
    manager = get_codeforces_r1_dataset(problems_ids, path_to_contest_data, path_to_test_files_data)

    passes_r1_tests = []
    is_problem_usable = []

    for _, submission in tqdm(list(dataset.iterrows())):
        problem_id = submission['problem_id']
        source_code =  base64.b64decode(submission['sourceCode']).decode('utf-8')
        programming_language = submission['programmingLanguage']

        problem_usable = manager.is_problem_usable(problem_id)
        is_problem_usable.append(problem_usable)
        if problem_usable:
            passes_r1_tests.append(await manager.evaluate_submission(problem_id, source_code, programming_language, 'hard'))
        else:
            passes_r1_tests.append(False)

    dataset['is_problem_usable'] = is_problem_usable
    dataset['passes_r1_tests'] = passes_r1_tests


def _create_dataset_per_user(user, submissions):
    logger.info(f"Computing anchor submissions for {user}...")

    submissions_df = [{"user": user,
                    "submissions_id": s["id"],
                   "problem_id": f"{s['problem']['contestId']}/{s['problem']['index']}",
                   "creationTimeSeconds":s['creationTimeSeconds'],
                   "programmingLanguage": s['programmingLanguage'],
                   "verdict":s["verdict"],
                   "sourceCode": s['sourceBase64']} for s in submissions]
    submissions_df = pd.DataFrame(submissions_df)

    submissions_df = submissions_df.groupby(['problem_id']).apply(_determine_anchor)
    submissions_df = submissions_df.set_index('submissions_id')

    return submissions_df

async def create_dataset(path_submissions:str, path_to_test_files_data:str, path_to_contest_data:str, output_path:str = "data/dataset.csv"):
    logger.info("Creating dataset...")

    submissions_dfs = []
    for submissions_path in glob(f"{path_submissions}/*.json"):
        with open(submissions_path, 'r', encoding='utf-8') as fp:
            submissions = json.load(fp)
        
        user_name = os.path.basename(submissions_path).split('_')
        user_name = "_".join(user_name[:-1])

        submissions_dfs.append(_create_dataset_per_user(user_name, submissions))

    submissions_dfs = pd.concat(submissions_dfs)

    await _append_submission_status_r1(submissions_dfs, path_to_test_files_data, path_to_contest_data)

    submissions_dfs.to_csv(output_path)