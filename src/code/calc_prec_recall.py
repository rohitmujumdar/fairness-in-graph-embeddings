import os

import numpy as np
import pandas as pd

# reco_df = pd.DataFrame(reco_score_data, columns=['userId',
#                                                  'movieId',
#                                                  'isRated',
#                                                  'predScore'])

RESULTS_DIR = "../results/strategy7"
reco_df = pd.read_csv(os.path.join(RESULTS_DIR, "reco.csv"), index_col=0)

grouped = reco_df.groupby(['userId'])
sorted_groups = []
for _, group in grouped:
    group = group.sort_values(by=['predScore'], ascending=False)
    sorted_groups.append(group)


def calc_precision_recall(sorted_groups, per=100):
    precision = []
    recall = []
    for group in sorted_groups:
        num_total_rated = (group['isRated'] == 1).sum()
        group = group.head(per)
        num_rated = (group['isRated'] == 1).sum()
        if per > 0 and num_total_rated > 0:
            precision.append(num_rated / per)
            recall.append(num_rated / num_total_rated)

    return np.mean(precision), np.mean(recall)


precision_list = []
recall_list = []
klist = [50, 100, 500, 1000]
for k in klist:
    precision, recall = calc_precision_recall(sorted_groups, k)
    print(f"Average precision @{k}: {precision}")
    print(f"Average recall @{k}: {recall}")
    precision_list.append(round(precision, 2))
    recall_list.append(round(recall, 2))

klist.insert(0, "k")
precision_list.insert(0, "Precision")
recall_list.insert(0, "Recall")
result_df = pd.DataFrame([precision_list, recall_list], columns=klist)
result_df.to_csv(os.path.join(RESULTS_DIR, "pr.csv"))
