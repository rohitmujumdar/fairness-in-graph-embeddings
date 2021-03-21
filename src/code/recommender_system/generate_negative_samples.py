import pandas as pd

data_file = '../../data/ml_df.pkl'
df_positive = pd.read_pickle(data_file)

user_ids = list(df_positive['userId'].unique())
movie_ids = list(df_positive['movieId'].unique())

user_df = df_positive[['userId', 'age', 'gender', 'occupation', 'zip code']]
user_df = user_df.groupby(['userId']).first().reset_index()

all_possible_edges = []
for user_id in user_ids:
    for movie_id in movie_ids:
        all_possible_edges.append([user_id, movie_id])
df_all_edges = pd.DataFrame(all_possible_edges, columns=['userId', 'movieId'])

df_all_edges = df_all_edges.merge(user_df,
                                  how='inner',
                                  left_on=['userId'],
                                  right_on=['userId'])

assert len(user_ids) == len(df_all_edges['userId'].unique())
assert len(movie_ids) == len(df_all_edges['movieId'].unique())
assert len(user_ids) * len(movie_ids) == len(df_all_edges)

merged_df = df_all_edges.merge(df_positive[['userId', 'movieId', 'rating']],
                               how='left',
                               left_on=['userId', 'movieId'],
                               right_on=['userId', 'movieId'])

assert len(merged_df[~merged_df['rating'].isnull()]) == len(df_positive)

df_negative = merged_df[merged_df['rating'].isnull()].copy()
df_negative['rating'] = 0

assert len(df_negative) + len(df_positive) == len(df_all_edges)
