import pandas as pd
from networkx import *

df_train = pd.read_csv('../data/df_train.csv', index_col=0)
df_train = df_train[df_train['rating'] > 0].copy()
user_df = df_train[['userId', 'age', 'gender', 'occupation', 'zip code']]
user_df = user_df.groupby(['userId']).first().reset_index()

G = nx.Graph()

# Add nodes
G.add_nodes_from(df_train.userId, bipartite=0)
G.add_nodes_from(df_train.movieId, bipartite=1)
usr_attrs = user_df.set_index('userId').to_dict('index')

nx.set_node_attributes(G, usr_attrs)

# Add weights for edges
G.add_weighted_edges_from([(uId, mId, rating) for (uId, mId, rating) in
                           df_train[
                               ['userId', 'movieId', 'rating']].to_numpy()])

print(G.is_directed(), G.is_multigraph(), is_bipartite(G))
nx.write_gpickle(G, "../data/train_graph.gpickle")