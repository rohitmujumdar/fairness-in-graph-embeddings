import networkx as nx
import pandas as pd
from gensim.models import KeyedVectors

embeddings_path = '../../data/ml_node2vec_model.embeddings'
data_file = '../../data/ml_df.pkl'
graph_path = '../../data/ml_graph.gpickle'

wv = KeyedVectors.load(embeddings_path, mmap='r')

df = pd.read_pickle(data_file)
print(len(df))

G = nx.read_gpickle(graph_path)


