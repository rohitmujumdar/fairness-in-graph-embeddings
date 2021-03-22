import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import pickle

# model_path = '../data/node2vec.model'
# model = Word2Vec.load(model_path)


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
    y[y > 0] = 1

    X = user_matrix * movie_matrix
    X_norm = np.linalg.norm(X, axis=1, ord=2, keepdims=True)
    X /= X_norm
    print(X.shape, y.shape)
    return X, y, keys


if __name__ == '__main__':
    embeddings_path = '../data/node2vec.embeddings'
    wv = KeyedVectors.load(embeddings_path, mmap='r')

    df_train = pd.read_csv('../data/df_train.csv', index_col=0)
    df_test = pd.read_csv('../data/df_test.csv', index_col=0)

    Xtrain, ytrain, train_keys = convert_to_classifier_input(df_train)
    Xtest, ytest, test_keys = convert_to_classifier_input(df_test)

    with open('../data/train_vectors.npy', 'wb') as f:
        np.save(f, Xtrain)
        np.save(f, ytrain)
    with open('../data/train_keys.pkl', 'wb') as f:
        pickle.dump(train_keys, f)

    with open('../data/test_vectors.npy', 'wb') as f:
        np.save(f, Xtest)
        np.save(f, ytest)

    with open('../data/test_keys.pkl', 'wb') as f:
        pickle.dump(test_keys, f)
