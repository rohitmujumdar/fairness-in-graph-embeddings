import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

with open('../data/train_vectors.npy', 'rb') as f:
    Xtrain = np.load(f)
    ytrain = np.load(f)

with open('../data/test_vectors.npy', 'rb') as f:
    Xtest = np.load(f)
    ytest = np.load(f)

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
print(classification_report(ytest, ypred))
