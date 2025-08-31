import datasets
import requests
from tqdm import tqdm
import time
def get_submissions_user(contest_id):
    idx = 1
    count = 10000
    submissions = []
    
    while True:
        fail_count = 0
        while fail_count < 3:
            try:
                time.sleep(2)
                response = requests.get(f'https://codeforces.com/api/contest.status?contestId={contest_id}&from={idx}&count={count}').json()
                submissions_view = response['result']
            except:
                print(f'Retry {fail_count + 1}/3')
                fail_count+=1
                continue
            break

        if fail_count == 3:
            print(f'Round skipped or partially done {contest_id}')
            break
        
        submissions.extend(submissions_view)
        if len(submissions_view) < count:
            break
        else:
            idx+=count
        
    
    return submissions

# submissions = get_submissions_user(100)

# submissions = submissions



ds = datasets.load_dataset("data/codeforces/submissions")

groups = {}

for idx, x in tqdm(list(enumerate(zip(ds['train']['contestId'], ds['train']['submission_id'])))):
    contest_id, submission_id = x
    if contest_id not in groups:
        groups[contest_id] = []
    
    groups[contest_id].append((idx, submission_id))

progress_bar = tqdm(list(groups.items()))

total_miss_hits = 0
total_hits = 0

all_handles = {idx: None for idx in range(len(ds['train']))}
count = 0
for g_id, submissions in progress_bar:
    real_submissions = get_submissions_user(g_id)

    match_index_to_id = {id_: idx for idx, id_ in submissions}

    matching = {s[1]:None for s in submissions}

    for rs in real_submissions:
        m_id = str(rs['id'])
        if m_id in matching:
            author = rs['author']
            if 'teamId' in author:
                matching[m_id] = author['teamId']
            else:
                matching[m_id] = author['participantId']
            total_hits+=1
        else:
            total_miss_hits+=1
    progress_bar.set_postfix({"miss": total_miss_hits, "hits":total_hits})

    matching_result = {match_index_to_id[k]: v for k,v in matching.items()}

    all_handles.update(matching_result)

    count+=1

ds = ds['train'].add_column('participantId', [v for _, v in sorted(list(all_handles.items()), key = lambda x: x[0])])

ds.save_to_disk('data/codeforces/submissions_extended')