import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

with open('../data/train_vectors.npy', 'rb') as f:
    Xtrain = np.load(f)
    ytrain = np.load(f)

with open('../data/test_vectors.npy', 'rb') as f:
    Xtest = np.load(f)
    ytest = np.load(f)

with open('../data/test_keys.pkl', 'rb') as f:
    test_keys = pickle.load(f)

np.random.seed(0)
indices = np.random.choice(len(Xtrain), size=len(Xtrain), replace=False)
Xtrain = Xtrain[indices]
ytrain = ytrain[indices]

clf = LogisticRegression(random_state=0)
clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtrain)
print("Train metrics")
print(classification_report(ytrain, ypred))

print("Test metrics")
ypred = clf.predict(Xtest)
ypred_prob = clf.predict_proba(Xtest)[:, 1]
print(classification_report(ytest, ypred))

reco_score_data = []
for idx, (user, movie) in enumerate(test_keys):
    reco_score_data.append(([user, movie, ytest[idx], ypred_prob[idx]]))
reco_df = pd.DataFrame(reco_score_data, columns=['userId',
                                                 'movieId',
                                                 'isRated',
                                                 'predScore'])

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
        precision.append(num_rated / per)
        recall.append(num_rated / num_total_rated)

    return np.mean(precision), np.mean(recall)


precision_list = []
recall_list = []
klist = []
k = 1
for i in range(1, 5):
    k *= 10
    precision, recall = calc_precision_recall(sorted_groups, k)
    print(f"Average precision @{k}: {precision}")
    print(f"Average recall @{k}: {recall}")
    klist.append(k)
    precision_list.append(round(precision, 2))
    recall_list.append(round(recall, 2))

klist.insert(0, "k")
precision_list.insert(0, "Precision")
recall_list.insert(0, "Recall")
result_df = pd.DataFrame([precision_list, recall_list], columns=klist)
result_df.to_csv('../results/node2vec_reco_results.csv')
