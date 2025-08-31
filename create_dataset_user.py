import datasets
import difflib
import pandas as pd

# submissions_ds = datasets.load_from_disk("data/codeforces/submissions_extended").to_pandas()

# submissions_ds = submissions_ds[submissions_ds['programmingLanguage'].str.contains('C++', regex=False) | 
#                                 submissions_ds['programmingLanguage'].str.contains('GNU C', regex=False)]


# submissions_ds = submissions_ds[~submissions_ds['participantId'].isna()]


# print(submissions_ds)

#for s in submissions_ds:
import time
import requests
import hashlib

CODEFORCES_API_KEY = ''
CODEFORCES_API_SECRET = ''

def get_submissions_user(user_id):
    idx = 1
    count = 10000
    submissions = []
    
    while True:
        fail_count = 0
        while fail_count < 3:
            try:
                time.sleep(2)
                my_time = time.time_ns() // 1000000000
                
                hsh = f'x1y1x2/user.status?apiKey={CODEFORCES_API_KEY}&count={count}&from={idx}&handle={user_id}&includeSources=true&time={my_time}#{CODEFORCES_API_SECRET}'
                hsh = hashlib.sha512(hsh.encode('utf-8')).hexdigest()
                response = requests.get(f'https://codeforces.com/api/user.status?apiKey={CODEFORCES_API_KEY}&count={count}&from={idx}&handle={user_id}&includeSources=true&time={my_time}&apiSig=x1y1x2{hsh}').json()
                submissions_view = response['result']
            except:
                print(f'Retry {fail_count + 1}/3')
                fail_count+=1
                continue
            break

        submissions.extend(submissions_view)
        if len(submissions_view) < count:
            break
        else:
            idx+=count
        
    
    return submissions

import base64

submissions = get_submissions_user('Arkin')
submissions = submissions

# problems_ds = datasets.load_dataset("data/codeforces/data")['test']
# generated_tests = datasets.load_dataset("data/codeforces/generated_tests")['test']

# all_contests = set(generated_tests['contestid'])

# submissions = [s for s in submissions if str(s['contestId']) in all_contests]

submissions_df = [{"submissions_id": s["id"],
                   "problem_id": f"{s['problem']['contestId']}/{s['problem']['index']}",
                   "creationTimeSeconds":s['creationTimeSeconds'],
                   "programmingLanguage": s['programmingLanguage'],
                   "verdict":s["verdict"],
                   "sourceCode": s['sourceBase64']} for s in submissions]

submissions_df = pd.DataFrame(submissions_df)

def determine_anchor(group):
    group_creation_dates = [(idx, e['verdict'], e['creationTimeSeconds']) for idx, e in group.iterrows()]
    anchors = [-1 for _, g in group.iterrows()]

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


submissions_df = submissions_df.groupby(['problem_id']).apply(determine_anchor)

def preprocess(s:str):
    return s.replace('\t', '')

def count_differences(source_a:str, source_b:str):
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

submissions_df = submissions_df.set_index('submissions_id')

number_additions = []
number_deletions = []

for _, s in submissions_df.iterrows():
    if s['AcceptedAnchor'] == -1:
        number_additions.append(-1)
        number_deletions.append(-1)
        continue

    s_sc = base64.b64decode(s['sourceCode']).decode('utf-8')
    t_sc = base64.b64decode(submissions_df.loc[s['AcceptedAnchor']]['sourceCode']).decode('utf-8')

    additions, deletions = count_differences(s_sc, t_sc)

    number_additions.append(additions)
    number_deletions.append(deletions)


submissions_df['code_additions'] = number_additions
submissions_df['code_deletions'] = number_deletions


submissions_df.to_csv('dataset.csv')


# problem_ds = problems_ds

# submissions_ds = datasets.load_dataset("data/codeforces/submissions")
# # generated_tests_ds = datasets.load_dataset("data/codeforces/generated_tests")

# for s in submissions_ds['train']:
#     if 'C++' in s['programmingLanguage']:
#         first_submission = s
#         break

# source = first_submission['source']
# problem_id = first_submission['problem_id']

# problem_info = [p for p in problems_ds if p['id'] == problem_id][0]

# tests = problem_info['official_tests']

# t(source, problem_info, tests)

# print(problems_ds)
# print(submissions_ds)
# print(generated_tests_ds)