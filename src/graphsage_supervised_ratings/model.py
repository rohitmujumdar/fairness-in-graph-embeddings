import torch
import torch.nn as nn
from torch.nn import init
import pandas as pd

import os
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
import pickle
import json

from encoders import Encoder
from aggregators import MeanAggregator


CLASS_NUM = 2
LIKE_THRESHOLD = 3
DO_FAIRNESS = False


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        # self.xent = nn.BCELoss(reduction='mean')  # reduction='none'
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim * 2))
        init.xavier_uniform_(self.weight)
        self.m = nn.Softmax()

    def forward(self, movie_nodes, user_nodes):
        movie_embeds = self.enc(movie_nodes, fairness = DO_FAIRNESS)
        user_embeds = self.enc(user_nodes)
        merge_movie_user = torch.cat((movie_embeds, user_embeds), 0)
        scores = self.weight.mm(merge_movie_user)
        scores = scores.t()
        # print(self.m(scores.data))
        return movie_embeds, user_embeds, scores

    def loss(self, movie_nodes, user_nodes, labels):
        movie_embeds, user_embeds, scores = self.forward(movie_nodes, user_nodes)
        # print(self.xent(scores.cuda(), torch.from_numpy(labels).cuda()).item())
        return self.xent(scores.cuda(), torch.from_numpy(labels).cuda())
        # return self.xent(torch.sigmoid(scores.cuda()), torch.FloatTensor(labels.squeeze()).cuda())

    def embeddings(self, node_id):
        return self.enc(node_id)

def load_MovieLensv2_test(user_idu, movie_idm, num_users, num_movies):

    print(num_users, num_movies)

    df_test = pd.read_csv('../df_test.csv', index_col=0)
    df_test = df_test[df_test['rating'] > 0].copy()

    adj_list_test = defaultdict(set)
    rating_list_test = defaultdict()

    for index, row in df_test.iterrows():
        # print(row['userId'], row['movieId'])

        if (row['userId'] in user_idu) and (row['movieId'] in movie_idm):

            user = user_idu[row['userId']]
            movie = movie_idm[row['movieId']]
            rating = row['rating']

            adj_list_test[user + num_movies].add(movie)
            adj_list_test[movie].add(user + num_movies)
            rating_list_test[(user + num_movies, movie)] = 1 if rating >= LIKE_THRESHOLD else 0


    # exit()
    # import pdb; pdb.set_trace()

    return rating_list_test, adj_list_test

def load_MovieLensv2_train():

    # Loads only train data

    df_train = pd.read_csv('../df_train.csv', index_col=0)
    df_train = df_train[df_train['rating'] > 0].copy()

    user_df = df_train[['userId', 'gender']]
    user_df = user_df.groupby(['userId']).first().reset_index()

    movie_df = df_train[['movieId']]
    movie_df = movie_df.groupby(['movieId']).first().reset_index()

    adj_list = defaultdict(set)
    rating_list = defaultdict()

    # get dicts
    user_idu = {}
    idu_user = {}
    user_gender_dict = {}

    user_ct = 0
    for index, row in user_df.iterrows():
        user_idu[row['userId']] = user_ct
        idu_user[user_ct] = row['userId']
        user_ct += 1
        
    movie_idm = {}
    idm_movie = {}
        
    movie_ct = 0
    for index, row in movie_df.iterrows():
        movie_idm[row['movieId']] = movie_ct
        idm_movie[movie_ct] = row['movieId']
        movie_ct += 1

    # get adj list
    for index, row in df_train.iterrows():
        # print(row['userId'], row['movieId'])

        user = user_idu[row['userId']]
        movie = movie_idm[row['movieId']]
        rating = row['rating']

        adj_list[user + movie_ct].add(movie)
        adj_list[movie].add(user + movie_ct)
        rating_list[(user + movie_ct, movie)] = 1 if rating >= LIKE_THRESHOLD else 0

    if DO_FAIRNESS:
        user_ct = 0
        for index, row in user_df.iterrows():
            user_gender_dict[user_ct + movie_ct] = row['gender']
            user_ct += 1



    num_nodes = user_ct + movie_ct
    feat_data = np.eye(num_nodes)

    print(user_ct, movie_ct)

    return feat_data, rating_list, adj_list, num_nodes, user_ct, movie_ct, user_idu, movie_idm, user_gender_dict




def run_movielens():
    np.random.seed(1)
    random.seed(1)
    # feat_data, rating_list, adj_list = load_movielens()
    feat_data_train, rating_list_train, adj_list_train, num_nodes_train, num_users_train, num_movies_train, user_dict_train, movie_dict_train, user_gender_dict = load_MovieLensv2_train()

    features = nn.Embedding(num_nodes_train, num_nodes_train)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data_train), requires_grad=True)
    features.cuda()

    if DO_FAIRNESS:
        agg1 = MeanAggregator(features, cuda=True, gender_map = user_gender_dict)
        enc1 = Encoder(features, num_nodes_train, 100, adj_list_train, agg1, gcn=True, cuda=True)

        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True, gender_map = user_gender_dict)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 100, adj_list_train, agg2,
                    base_model=enc1, gcn=True, cuda=True)
    else:
        agg1 = MeanAggregator(features, cuda=True)
        enc1 = Encoder(features, num_nodes_train, 100, adj_list_train, agg1, gcn=True, cuda=True)

        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 100, adj_list_train, agg2,
                    base_model=enc1, gcn=True, cuda=True)

    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(CLASS_NUM, enc2)
    graphsage.cuda()
    rand_indices = np.random.permutation(num_movies_train)

    train = list(rand_indices[:num_movies_train])  # train list contains the movie nodes

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.01)
    for batch in range(50000):

        if batch % 100 == 0:
            print(batch)

        batch_nodes = train[:1]  # movie node
        random.shuffle(train)
        user_node = random.sample(adj_list_train[batch_nodes[0]], 1) # user node

        optimizer.zero_grad()
        new_lables = [rating_list_train[(user_node[0], batch_nodes[0])]]
        # for i in batch_nodes:
        #     new_lables.append(labels[i])

        loss = graphsage.loss(batch_nodes, user_node,
                              np.array(new_lables))
        loss.backward()
        optimizer.step()

    if DO_FAIRNESS:
        torch.save(graphsage.state_dict(), "./models/graphsage_ratings_fair_model")
    else:
        torch.save(graphsage.state_dict(), "./models/graphsage_ratings_model")



def accuracy():
    feat_data_train, rating_list_train, adj_list_train, num_nodes_train, num_users_train, num_movies_train, user_dict_train, movie_dict_train, user_gender_dict = load_MovieLensv2_train()

    rating_list_test, adj_list_test = load_MovieLensv2_test(user_dict_train, movie_dict_train, num_users_train, num_movies_train)
    
    features = nn.Embedding(num_nodes_train, num_nodes_train)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data_train), requires_grad=True)
    features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, num_nodes_train, 100, adj_list_train, agg1, gcn=True, cuda=True)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 100, adj_list_train, agg2,
                   base_model=enc1, gcn=True, cuda=True)

    if DO_FAIRNESS:
        print("doing fair training")
        agg1 = MeanAggregator(features, cuda=True, gender_map = user_gender_dict)
        enc1 = Encoder(features, num_nodes_train, 100, adj_list_train, agg1, gcn=True, cuda=True)

        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True, gender_map = user_gender_dict)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 100, adj_list_train, agg2,
                    base_model=enc1, gcn=True, cuda=True)
    else:
        agg1 = MeanAggregator(features, cuda=True)
        enc1 = Encoder(features, num_nodes_train, 100, adj_list_train, agg1, gcn=True, cuda=True)

        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 100, adj_list_train, agg2,
                    base_model=enc1, gcn=True, cuda=True)

    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(CLASS_NUM, enc2)
    graphsage.cuda()
    graphsage.load_state_dict(torch.load("./models/graphsage_ratings_model"))
    total = 0
    acc = 0

    # import pdb; pdb.set_trace()

    # for user_id in range(num_movies_train + 1, num_movies_train + 1):  # user_nodes+MOVIE_NUM

    for user_id in range(num_movies_train, num_nodes_train):
        for movie_id in adj_list_test[user_id]:
            movie_embeds, user_embeds, scores = graphsage.forward([movie_id], [user_id])
            scores = torch.softmax(scores, dim=-1)
            pred_label = np.argmax(scores.cpu().detach().numpy())

            if (user_id, movie_id) not in rating_list_test:
                print(user_id, movie_id)
                continue

            gdtruth_label = rating_list_test[(user_id, movie_id)]
            total += 1
            if pred_label == gdtruth_label:
                acc += 1

            # if total > 10000:
            #     break
            
        # if total > 10000:
        #         break

    print(acc/total * 1.0)


if __name__ == "__main__":
    # accuracy()
    x = torch.randn(5).cuda()
    print(x)
    run_movielens()
    # load_movielens()
    accuracy()