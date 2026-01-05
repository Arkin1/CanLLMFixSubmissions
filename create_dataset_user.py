import pandas as pd
import time
import requests
import hashlib
import base64
from utils import count_differences, RetryExecution
import logging
import sys
import os

logger = logging.getLogger()

def get_submissions_user(user_id, codeforces_api_key, codeforces_api_secret):
    idx = 1
    count = 10000
    submissions = []

    with RetryExecution("GetSubmissionsUser", 3, logger) as retry:
        while True:
            def get_submission_view():
                time.sleep(2)
                my_time = time.time_ns() // 1000000000
                
                hsh = f'x1y1x2/user.status?apiKey={codeforces_api_key}&count={count}&from={idx}&handle={user_id}&includeSources=true&time={my_time}#{codeforces_api_secret}'
                hsh = hashlib.sha512(hsh.encode('utf-8')).hexdigest()
                response = requests.get(f'https://codeforces.com/api/user.status?apiKey={codeforces_api_key}&count={count}&from={idx}&handle={user_id}&includeSources=true&time={my_time}&apiSig=x1y1x2{hsh}')
                response.raise_for_status()

                response = response.json()
                submissions_view = response['result']
                return submissions_view
            
            submissions_view = retry(get_submission_view)
            submissions.extend(submissions_view)
            if len(submissions_view) < count:
                break
            else:
                idx+=count
    
    return submissions

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

def create_dataset(user, submissions):
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

    return submissions_df


# if __name__ == '__main__':
#     codeforces_api_key = os.getenv('CODEFORCES_API_KEY')
#     codeforces_api_secret = os.getenv('CODEFORCES_API_SECRET')

#     handler = logging.StreamHandler(sys.stdout)
#     handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

#     submissions = get_submissions_user('Arkin', codeforces_api_key, codeforces_api_secret)
#     submissions_df = create_dataset('Arkin', submissions)

#     submissions_df.to_csv('dataset.csv')
