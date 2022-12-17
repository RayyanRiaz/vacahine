import itertools
import math
import os
import pickle
import random
from enum import Enum, auto

import dgl
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from torch_geometric.utils import remove_self_loops, add_self_loops, negative_sampling



EPS = 1e-15
MAX_LOGSTD = 10

class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class Scores(AutoName):
    NMI_KM = auto()
    ARI_KM = auto()
    NMI_Pc = auto()
    ARI_Pc = auto()
    NMI_Qc = auto()
    ARI_Qc = auto()
    # NODE_CLASSIFICATION_ACCURACY = auto()
    F1_MICRO = auto()
    F1_MACRO = auto()



class Losses(AutoName):
    L_Total = auto()
    L_Recon = auto()
    L_KL_p_and_qc = auto()
    L_Enc = auto()
    L_KMeans_logits = auto()
    L_DGI_subgraphs = auto()



def map_labels(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return ind


def predict_node_classification(train_z, train_y, test_z,
                                # solver='lbfgs',
                                solver='liblinear',
                                multi_class='auto', *args, **kwargs):
    clf = LogisticRegression(solver=solver, multi_class=multi_class, *args, **kwargs) \
        .fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())

    return clf.predict(test_z.detach().cpu().numpy()).astype(int)


def kv_to_print_str(kvs, remove_str="Scores."):
    return "".join(["{}: {:.4f}||\t".format(str(k).replace(remove_str, ""), v) for k, v in kvs.items()])


def edge_type_and_index(dataset):
    edge_index = list()
    edge_type = list()
    edge_type_counter = 0
    for key in dataset['edge_index_dict'].keys():
        edge_index.append(dataset['edge_index_dict'][key])
        edge_type = edge_type + [edge_type_counter] * dataset['edge_index_dict'][key].shape[1]
        edge_type_counter += 1
    edge_index = torch.cat(edge_index, dim=1)
    edge_type = torch.tensor(edge_type)
    return edge_index, edge_type


def nshe_sample_metapath_instances(edge_index_dict, metapath, samples_per_starting_node=10):
    # dgl.random.seed(seed)
    # dgl.seed(seed)
    dgl_data_dict = {(k[0], k[0] + k[1], k[1]): v.T.tolist() for k, v in edge_index_dict.items()}
    dgl_heterograph = dgl.heterograph(dgl_data_dict)
    samples = []
    for i in range(samples_per_starting_node):
        starting_nodes_sorted = sorted(set(edge_index_dict[(list(metapath[:2])[0], list(metapath[:2])[1])].tolist()[0]))
        rw_sample = dgl.sampling.random_walk(g=dgl_heterograph, metapath=[metapath[i:i + 2] for i in range(len(metapath) - 1)],
                                             nodes=starting_nodes_sorted)[0]
        assert len(rw_sample) == len(starting_nodes_sorted)
        samples.append(rw_sample)

    samples = [s for k in samples for s in k]
    samples = sorted(samples, key=lambda s: s[0])
    return samples


def generate_samples(edge_index_dict, metapaths, nodes_per_metapath, save_name=None, return_format="tensor"):
    assert return_format in ["list", "tensor"]
    if type(metapaths) is list:
        samples = []
        for mp, n in zip(metapaths, nodes_per_metapath):
            mp_samples = nshe_sample_metapath_instances(edge_index_dict, metapath=mp, samples_per_starting_node=n)
            samples.append(torch.cat([s[None, :] for s in mp_samples], dim=0))
        if return_format == "tensor":
            samples = list(itertools.chain(*samples))
            samples = sorted(samples, key=lambda x: x[0])
    else:
        samples = nshe_sample_metapath_instances(edge_index_dict, metapath=metapaths, samples_per_starting_node=nodes_per_metapath)

    if return_format == "tensor":
        samples = torch.cat([s[None, :] for s in samples], dim=0)

    if save_name is not None:
        with open(save_name, "wb") as f:
            pickle.dump(samples, f)
    return samples

def generate_samples_new(edge_index_dict, metapaths, nodes_per_metapath, save_name=None):
    samples = {}
    if type(metapaths) is list:
        for mp, n in zip(metapaths, nodes_per_metapath):
            if not int(mp[0]) in samples:
                samples[int(mp[0])] = {}
            samples[int(mp[0])][mp] = {"N": n}
            samples[int(mp[0])][mp]["samples"] = nshe_sample_metapath_instances(edge_index_dict, metapath=mp, samples_per_starting_node=n)
            samples[int(mp[0])][mp]["samples"] = torch.cat([s[None, :] for s in samples[int(mp[0])][mp]["samples"]], dim=0)
    else:
        samples = nshe_sample_metapath_instances(edge_index_dict, metapath=metapaths, samples_per_starting_node=nodes_per_metapath)
        samples = torch.cat([s[None, :] for s in samples], dim=0)

    if save_name is not None:
        with open(save_name, "wb") as f:
            pickle.dump(samples, f)
    return samples

def get_negative_edge_index(pos_edge_index, N, num_neg_samples=None):
    pos_edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index, _ = add_self_loops(pos_edge_index)
    neg_edge_index = negative_sampling(pos_edge_index, N, num_neg_samples)
    return neg_edge_index


inf = math.inf
def kl_categorical_categorical(p, q):
    t = p.probs * (p.logits - q.logits)
    t[(q.probs == 0).expand_as(t)] = inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.clamp(max=MAX_LOGSTD).sum(-1)

