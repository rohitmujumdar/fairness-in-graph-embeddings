import os
import pickle

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


def convert_to_classifier_input(df):
    user_vectors = []
    movie_vectors = []
    keys = []
    y = []
    for it, row in df.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        try:
            movie_vectors.append(wv[movie_id])
            user_vectors.append(wv[user_id])
            keys.append((user_id, movie_id))
            y.append(row['rating'])
        except KeyError as e:
            pass

    user_matrix = np.array(user_vectors)
    movie_matrix = np.array(movie_vectors)
    y = np.array(y)
    y[y <= 3] = 0
    y[y > 3] = 1

    X = np.concatenate((user_matrix, movie_matrix), axis=1)
    # X = user_matrix * movie_matrix
    # X_norm = np.linalg.norm(X, axis=1, ord=2, keepdims=True)
    # X /= X_norm
    print(X.shape, y.shape)
    return X, y, keys


if __name__ == '__main__':
    FEATURE_DIR = "../data/features/strategy7"
    embeddings_path = os.path.join(FEATURE_DIR, 'word2vec.embeddings')

    wv = KeyedVectors.load(embeddings_path, mmap='r')

    df_train = pd.read_csv('../data/df_train.csv', index_col=0)
    df_train = df_train[df_train['rating'] > 0]

    df_test = pd.read_csv('../data/df_test_20k.csv', index_col=0)

    Xtrain, ytrain, train_keys = convert_to_classifier_input(df_train)
    Xtest, ytest, test_keys = convert_to_classifier_input(df_test)

    with open(os.path.join(FEATURE_DIR, "train_vectors.npy"), 'wb') as f:
        np.save(f, Xtrain)
        np.save(f, ytrain)

    with open(os.path.join(FEATURE_DIR, "train_keys.pkl"), 'wb') as f:
        pickle.dump(train_keys, f)

    with open(os.path.join(FEATURE_DIR, "test_vectors_20k.npy"), 'wb') as f:
        np.save(f, Xtest)
        np.save(f, ytest)

    with open(os.path.join(FEATURE_DIR, "test_keys_20k.pkl"), 'wb') as f:
        pickle.dump(test_keys, f)
