import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

FEATURE_DIR = "../data/features/strategy7"
RESULTS_DIR = "../results/strategy7"

with open(os.path.join(FEATURE_DIR, "train_vectors.npy"), 'rb') as f:
    Xtrain = np.load(f)
    ytrain = np.load(f)

with open(os.path.join(FEATURE_DIR, "test_vectors_20k.npy"), 'rb') as f:
    Xtest = np.load(f)
    ytest = np.load(f)

with open(os.path.join(FEATURE_DIR, "test_keys_20k.pkl"), 'rb') as f:
    test_keys = pickle.load(f)

np.random.seed(0)
# Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.2,
#                                               random_state=0)

indices = np.random.choice(len(Xtrain), size=len(Xtrain), replace=False)
Xtrain = Xtrain[indices]
ytrain = ytrain[indices]

clf = LogisticRegression(random_state=0, class_weight='balanced', C=1)
# clf = SVC(random_state=0, C=1)
clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtrain)
print("Train metrics")
print(classification_report(ytrain, ypred))

print("Test metrics")
ypred = clf.predict(Xtest)
ypred_prob = clf.predict_proba(Xtest)[:, 1]
# ypred_prob = clf.decision_function(Xtest)
print(classification_report(ytest, ypred))

# reco_score_data = []
# for idx, (user, movie) in enumerate(test_keys):
#     reco_score_data.append(([user, movie, ytest[idx], ypred_prob[idx]]))
# reco_df = pd.DataFrame(reco_score_data, columns=['userId',
#                                                  'movieId',
#                                                  'isRated',
#                                                  'predScore'])
#
# reco_df.to_csv(os.path.join(RESULTS_DIR, "reco.csv"))
