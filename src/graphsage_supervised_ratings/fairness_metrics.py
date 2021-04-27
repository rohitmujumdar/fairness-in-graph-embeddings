import os

import networkx as nx
import pandas as pd
import pickle

G = nx.read_gpickle("./ml_graph_with_movies (1).gpickle")
RESULTS_DIR = './results/'
# RESULTS_DIR = './results_fair/'

# idu_user = pickle.load(open( "idu_user.p", "rb" ))
# idm_movie = pickle.load(open( "idm_movie.p", "rb" ))


genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
          'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
genders = ['M', 'F']

movie_genre = []
user_gender_map = {}

for node in G.nodes():
    if str.startswith(node, 'movie'):
        for key, val in G.nodes[node].items():
            if key in genres and val == 1:
                movie_genre.append([node, key])

    if str.startswith(node, 'user'):
        user_gender_map[node] = G.nodes[node]['gender']

# TODO: load properly from json dict and cast to pandas
reco_df = pd.read_csv(os.path.join(RESULTS_DIR, 'reco.csv'), index_col=0)

k = 200
grouped = reco_df.groupby(['userId'])
sorted_groups = []
for _, group in grouped:
    group = group.sort_values(by=['predScore'], ascending=False)
    sorted_groups.append(group.head(k))

reco_k = pd.concat(sorted_groups)

user_gender_df = pd.DataFrame.from_dict(user_gender_map, orient='index',
                                        columns=['gender']).reset_index()

user_gender_df.rename(columns={'index': "userId"}, inplace=True)

movie_genre_df = pd.DataFrame(movie_genre, columns=['movieId', 'genre'])

reco_k = reco_k.merge(user_gender_df, left_on='userId', right_on='userId',
                      how='inner')
reco_k = reco_k.merge(movie_genre_df, left_on='movieId', right_on='movieId',
                      how='inner')

gender_count = user_gender_df.groupby(['gender']).size()
genre_count = movie_genre_df.groupby(['genre']).size()

gender_count.to_csv(os.path.join(RESULTS_DIR, 'reco_group.csv'))
genre_count.to_csv(os.path.join(RESULTS_DIR, 'reco_group.csv'))


def get_max_count(row):
    return gender_count[row['gender']] * genre_count[row['genre']]


reco_group = reco_k.groupby(['gender', 'genre']).size().reset_index().rename(
    columns={0: 'count'})

# reco_group['max_count'] = reco_group.apply(lambda x: get_max_count(x),
# axis=1)
# reco_group['fraction'] = reco_group['count'] / reco_group['max_count']
reco_group.to_csv(os.path.join(RESULTS_DIR, 'reco_group.csv'))
