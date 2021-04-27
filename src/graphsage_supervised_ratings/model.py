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


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc, is_fairmodel):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim * 2))
        init.xavier_uniform_(self.weight)

        self.m = nn.Softmax()
        self.is_fairmodel = is_fairmodel

    def forward(self, movie_nodes, user_nodes):
        movie_embeds = self.enc(movie_nodes, fairness = self.is_fairmodel)
        user_embeds = self.enc(user_nodes)
        
        merge_movie_user = torch.cat((movie_embeds, user_embeds), 0)
        scores = self.weight.mm(merge_movie_user)
        scores = scores.t()
        
        return movie_embeds, user_embeds, scores

    def loss(self, movie_nodes, user_nodes, labels):
        movie_embeds, user_embeds, scores = self.forward(movie_nodes, user_nodes)

        return self.xent(scores.cuda(), torch.from_numpy(labels).cuda())

    def embeddings(self, node_id):
        return self.enc(node_id)

def load_MovieLensv2_test(user_idu, movie_idm, num_users, num_movies):

    # Loads only test data
    print(num_users, num_movies)

    df_test = pd.read_csv('../df_test.csv', index_col=0)
    df_test = df_test[df_test['rating'] > 0].copy()

    adj_list_test = defaultdict(set)
    rating_list_test = defaultdict()

    for index, row in df_test.iterrows():
        if (row['userId'] in user_idu) and (row['movieId'] in movie_idm):

            user = user_idu[row['userId']]
            movie = movie_idm[row['movieId']]
            rating = row['rating']

            adj_list_test[user + num_movies].add(movie)
            adj_list_test[movie].add(user + num_movies)
            rating_list_test[(user + num_movies, movie)] = 1 if rating >= LIKE_THRESHOLD else 0

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
    movie_genre_dict = {}

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

    user_ct = 0
    for index, row in user_df.iterrows():
        user_gender_dict[user_ct + movie_ct] = row['gender']
        user_ct += 1

    num_nodes = user_ct + movie_ct
    feat_data = np.eye(num_nodes)

    pickle.dump(idu_user, open( "idu_user.p", "wb"))
    pickle.dump(idm_movie, open( "idm_movie.p", "wb"))

    print(user_ct, movie_ct)

    return feat_data, rating_list, adj_list, num_nodes, user_ct, movie_ct, user_idu, movie_idm, user_gender_dict, idu_user, idm_movie




def run_movielens(do_fairness=True):
    np.random.seed(1)
    random.seed(1)
    
    feat_data_train, rating_list_train, adj_list_train, num_nodes_train, num_users_train, num_movies_train, user_dict_train, movie_dict_train, user_gender_dict, idu_user, idm_movie = load_MovieLensv2_train()

    features = nn.Embedding(num_nodes_train, num_nodes_train)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data_train), requires_grad=True)
    features.cuda()

    if do_fairness:
        print("training with fair model")
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

    graphsage = SupervisedGraphSage(CLASS_NUM, enc2, is_fairmodel=do_fairness)
    graphsage.cuda()
    rand_indices = np.random.permutation(num_movies_train)

    train = list(rand_indices[:num_movies_train])  # train list contains the movie nodes

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, graphsage.parameters()))
    for batch in range(20000):

        batch_nodes = train[:1000]  # movie node
        random.shuffle(train)
        user_nodes = []

        for bnode in batch_nodes:
            unode = random.sample(adj_list_train[bnode], 1)[0] # user node
            user_nodes.append(unode)

        optimizer.zero_grad()
        new_labels = []
        for k in range(len(batch_nodes)):
            new_labels.append(rating_list_train[(user_nodes[k], batch_nodes[k])])

        loss = graphsage.loss(batch_nodes, user_nodes,
                              np.array(new_labels))
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(batch, loss)


    if do_fairness:
        torch.save(graphsage.state_dict(), "./models/graphsage_ratings_fair_model_new")
    else:
        torch.save(graphsage.state_dict(), "./models/graphsage_ratings_model_new")



def accuracy(use_new = True, do_fairness=True):
    feat_data_train, rating_list_train, adj_list_train, num_nodes_train, num_users_train, num_movies_train, user_dict_train, movie_dict_train, user_gender_dict, idu_user, idm_movie = load_MovieLensv2_train()

    rating_list_test, adj_list_test = load_MovieLensv2_test(user_dict_train, movie_dict_train, num_users_train, num_movies_train)
    
    features = nn.Embedding(num_nodes_train, num_nodes_train)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data_train), requires_grad=True)
    features.cuda()

    if do_fairness:
        print("doing fair evaluation")
        agg1 = MeanAggregator(features, cuda=True, gender_map = user_gender_dict)
        enc1 = Encoder(features, num_nodes_train, 100, adj_list_train, agg1, gcn=True, cuda=True)

        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True, gender_map = user_gender_dict)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 100, adj_list_train, agg2,
                    base_model=enc1, gcn=True, cuda=True)
    else:
        print("doing normal evaluation")
        agg1 = MeanAggregator(features, cuda=True)
        enc1 = Encoder(features, num_nodes_train, 100, adj_list_train, agg1, gcn=True, cuda=True)

        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 100, adj_list_train, agg2,
                    base_model=enc1, gcn=True, cuda=True)

    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(CLASS_NUM, enc2, is_fairmodel=do_fairness)
    graphsage.cuda()

    if use_new:
        print("using new models")

        if do_fairness:
            print("loading fair model")
            graphsage.load_state_dict(torch.load("./models/graphsage_ratings_fair_model_new"))
        else:
            print("loading normal model")
            graphsage.load_state_dict(torch.load("./models/graphsage_ratings_model_new"))

    else:
        print("using old models")

        if do_fairness:
            print("loading fair model")
            graphsage.load_state_dict(torch.load("./models/graphsage_ratings_fair_model"))
        else:
            print("loading normal model")
            graphsage.load_state_dict(torch.load("./models/graphsage_ratings_model"))
        

    # slow version
    # total = 0
    # acc = 0

    # for user_id in range(num_movies_train, num_nodes_train):
    #     for movie_id in adj_list_test[user_id]:
            
    #         movie_embeds, user_embeds, scores = graphsage.forward([movie_id], [user_id])
    #         scores = torch.softmax(scores, dim=-1)
    #         print(scores.shape)
    #         pred_label = np.argmax(scores.cpu().detach().numpy(), axis=1)
    #         print(pred_label)
    #         import pdb; pdb.set_trace()

    #         if (user_id, movie_id) not in rating_list_test:
    #             print(user_id, movie_id)
    #             continue

    #         gdtruth_label = rating_list_test[(user_id, movie_id)]
    #         total += 1
    #         if pred_label == gdtruth_label:
    #             acc += 1

    # print(acc/total * 1.0)


    ## faster batched version
    total = 0
    acc = 0

    all_preds = []
    all_labels = []

    for user_id in range(num_movies_train, num_nodes_train):

        if user_id in adj_list_test:
            # print(user_id, total)

            movies = list(adj_list_test[user_id])
            users = [user_id for movie in movies]
                
            movie_embeds, user_embeds, scores = graphsage.forward(movies, users)
            scores = torch.softmax(scores, dim=-1)
            # print(scores.shape)

            pred_labels = np.argmax(scores.cpu().detach().numpy(), axis=1)
            # print(pred_labels)

            gt_labels = np.array([rating_list_test[(user_id, movie_id)] for movie_id in movies])
            # print(pred_labels)
            # print(gt_labels)
            # import pdb; pdb.set_trace()

            total += len(movies)
            acc += (gt_labels == pred_labels).sum()

            all_preds += list(pred_labels)
            all_labels += list(gt_labels)

    print(acc/total * 1.0)

    print("f1 score is")
    fs = f1_score(all_labels, all_preds)
    print(fs)


def save_scores(use_new, do_fairness):
    feat_data_train, rating_list_train, adj_list_train, num_nodes_train, num_users_train, num_movies_train, user_dict_train, movie_dict_train, user_gender_dict, idu_user, idm_movie = load_MovieLensv2_train()

    rating_list_test, adj_list_test = load_MovieLensv2_test(user_dict_train, movie_dict_train, num_users_train, num_movies_train)
    

    features = nn.Embedding(num_nodes_train, num_nodes_train)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data_train), requires_grad=True)
    features.cuda()

    if do_fairness:
        print("doing fair scores")
        agg1 = MeanAggregator(features, cuda=True, gender_map = user_gender_dict)
        enc1 = Encoder(features, num_nodes_train, 100, adj_list_train, agg1, gcn=True, cuda=True)

        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True, gender_map = user_gender_dict)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 100, adj_list_train, agg2,
                    base_model=enc1, gcn=True, cuda=True)
    else:
        print("doing normal scores")
        agg1 = MeanAggregator(features, cuda=True)
        enc1 = Encoder(features, num_nodes_train, 100, adj_list_train, agg1, gcn=True, cuda=True)

        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=True)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 100, adj_list_train, agg2,
                    base_model=enc1, gcn=True, cuda=True)

    enc1.num_samples = 10
    enc2.num_samples = 10

    # dict_result = {}
    temp_result = {}
    graphsage = SupervisedGraphSage(CLASS_NUM, enc2, is_fairmodel=do_fairness)
    graphsage.cuda()

    if use_new:
        print("using new models")

        if do_fairness:
            print("loading fair model")
            graphsage.load_state_dict(torch.load("./models/graphsage_ratings_fair_model_new"))
        else:
            print("loading normal model")
            graphsage.load_state_dict(torch.load("./models/graphsage_ratings_model_new"))

    else:
        print("using old models")

        if do_fairness:
            print("loading fair model")
            graphsage.load_state_dict(torch.load("./models/graphsage_ratings_fair_model"))
        else:
            print("loading normal model")
            graphsage.load_state_dict(torch.load("./models/graphsage_ratings_model"))


    #### Slow version ######
    # df = pd.DataFrame(columns=['userId', 'movieId', 'predScore'])
    # df_ct = 0
    # # TODO: make this fast via batches
    # for user_id in range(num_movies_train, num_nodes_train):

    #     # if user_id in adj_list_test:
    #         print(idu_user[user_id - num_movies_train])
    #         for movie_id in range(0, num_movies_train):

    #             movie_embeds, user_embeds, scores = graphsage.forward([movie_id], [user_id])
    #             scores = torch.softmax(scores, dim=-1)

    #             temp_result[idm_movie[movie_id]] = float(scores.cpu().detach().numpy()[:, -1][0])

    #             # print(idu_user[user_id - num_movies_train], idm_movie[movie_id], scores.cpu().detach().numpy()[:, -1][0])
    #             # if rating_list_test[(user_id, movie_id)] >= LIKE_THRESHOLD:
    #             #     is_rated = 1
    #             # else:
    #             #     is_rated = 0

    #             df.loc[df_ct] = [idu_user[user_id - num_movies_train], idm_movie[movie_id], scores.cpu().detach().numpy()[:, -1][0]]
    #             df_ct += 1


        # temp_result = sorted(temp_result.items(), key=lambda x: -x[1])[0:100]
        # dict_result[idu_user[user_id - num_movies_train]] = temp_result
        # temp_result = {}
    ########################

    ### Fast Version #######
    df = pd.DataFrame(columns=['userId', 'movieId', 'predScore'])
    df_ct = 0

    for user_id in range(num_movies_train, num_movies_train + 100):

        print(idu_user[user_id - num_movies_train])
        
        movies = [movie_id for movie_id in range(0, num_movies_train)]
        users = [user_id for movie_id in range(0, num_movies_train)]

        # for movie_id in range(0, num_movies_train):

        movie_embeds, user_embeds, scores = graphsage.forward(movies, users)
        scores = torch.softmax(scores, dim=-1)

        for idx, movie_id in enumerate(movies):
            # temp_result[idm_movie[movie_id]] = float(scores.cpu().detach().numpy()[:, -1][idx])

            df.loc[df_ct] = [idu_user[user_id - num_movies_train], idm_movie[movie_id], scores.cpu().detach().numpy()[:, -1][idx]]
            df_ct += 1
            # import pdb; pdb.set_trace()
        
    ###########################
    
    if do_fairness:
        df.to_csv('./results_fair/reco.csv')
    else:
        df.to_csv('./results/reco.csv')

    # with open('scores.json', 'w') as output:
    #     json.dump(dict_result, output)



if __name__ == "__main__":

    # load_MovieLensv2_train()
    # run_movielens(do_fairness=True)

    # accuracy(use_new=True, do_fairness=False)
    # accuracy(use_new=True, do_fairness=True)
    
    # save_scores(use_new=True, do_fairness=False)
    save_scores(use_new=True, do_fairness=True)

    # feat_data_train, rating_list_train, adj_list_train, num_nodes_train, num_users_train, num_movies_train, user_dict_train, movie_dict_train, user_gender_dict, idu_user, idm_movie = load_MovieLensv2_train()

    # rating_list_test, adj_list_test = load_MovieLensv2_test(user_dict_train, movie_dict_train, num_users_train, num_movies_train)