import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False, gender_map = None, fairness = False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.gender_map = gender_map
        
    def forward(self, nodes, to_neighs, num_sample=10, fairness = False):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set

        # TODO: change the sampling strategy
        # if self.movie_to_user:
        # import pdb; pdb.set_trace()


        if not num_sample is None:
            # we can change the function to sample the neighbors
            _sample = random.sample

            if (self.gender_map is not None) and fairness:
                # import pdb; pdb.set_trace()
                male_to_neighs = []
                female_to_neighs = []
                for to_neigh in to_neighs:
                    male_to_neigh = set()
                    female_to_neigh = set()
                    for neigh in to_neigh:
                        if neigh not in self.gender_map:
                            import pdb; pdb.set_trace()

                        gend_user = self.gender_map[neigh]
                        if gend_user == "M":
                            male_to_neigh.add(neigh)
                        else:
                            female_to_neigh.add(neigh)
                    male_to_neighs.append(male_to_neigh)
                    female_to_neighs.append(female_to_neigh)


                samp_neighs_male = [_set(_sample(male_to_neigh, int(num_sample/2),)) if len(male_to_neigh) >= int(num_sample/2) else male_to_neigh for male_to_neigh in male_to_neighs] 

                samp_neighs_female = [_set(_sample(female_to_neigh, int(num_sample/2),)) if len(female_to_neigh) >= int(num_sample/2) else female_to_neigh for female_to_neigh in female_to_neighs]

                # print("hi")
                samp_neighs = []
                for i in range(len(samp_neighs_male)):
                    new_neighs = samp_neighs_male[i] | samp_neighs_female[i]
                    samp_neighs.append(new_neighs) 

            else:
                samp_neighs = [_set(_sample(to_neigh, num_sample,)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        # also add the node itself to do the concatenation part
        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]

        # this line is so cool, concise
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]

        # mask matrix is the A matrix in our system, if we want to take the weight of edge in account,\
        # we can put the weight of edge in the A, then the A will become a weighted graph adjMatrix
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()

        # sum() 1 means the output should be in 1 dimension and true means the output should be in same dimension\
        # 4*4 -> 4*1
        num_neigh = mask.sum(1, keepdim=True)
        # divide corresponding to the D**-1
        mask = mask.div(num_neigh)

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        # corresponding to the D**-1 * A
        to_feats = mask.mm(embed_matrix)
        return to_feats