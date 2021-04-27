import networkx as nx
from node2vec import Node2Vec

# Import graph
graph = nx.read_gpickle("../data/train_graph.gpickle")

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH
# workers=1**
node2vec = Node2Vec(graph,
                    dimensions=64,
                    walk_length=30,
                    num_walks=20,
                    workers=4,
                    weight_key='weight',
                    p=.5,
                    q=.25)  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)
# Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and
# `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
# print(model.wv.most_similar('user1'))  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format('../data/node2vec.model')

# Save model for later use
model.save('../data/features/node2vec/word2vec.embeddings')
