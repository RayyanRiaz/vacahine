import glob
import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset, Data

from code.extras import edge_type_and_index


class N_Dataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        self.data_path = os.path.join(root, self.name)
        super(N_Dataset, self).__init__(self.data_path, transform, pre_transform)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def read_data_files(self):
        data_content = dict()
        for file in glob.glob(os.path.join(self.data_path, "*.txt")):
            try:
                data_content[file.split(os.sep)[-1].replace(".txt", "")] = np.loadtxt(file, dtype=int)
            except:
                data_content[file.split(os.sep)[-1].replace(".txt", "")] = np.loadtxt(file, dtype=str, delimiter="\t")
        return data_content

    def wrap_processing(self, data_list):
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class DBLP_N_P(N_Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(DBLP_N_P, self).__init__(root, "DBLP_N_P", transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data_content = self.read_data_files()

        node_type_mapping = {"a": "0", "p": "1", "c": "2"}
        node_id_mapping = {x[0]: int(x[1]) for x in data_content["node2id"]}

        relations = data_content['relations']
        edge_index_dict = {
            ('0', '1'): torch.from_numpy(relations[relations[:, 2] == 0][:, [0, 1]].T),
            ('1', '0'): torch.from_numpy(relations[relations[:, 2] == 0][:, [1, 0]].T),
            ('1', '2'): torch.from_numpy(relations[relations[:, 2] == 1][:, [0, 1]].T),
            ('2', '1'): torch.from_numpy(relations[relations[:, 2] == 1][:, [1, 0]].T)
        }
        node_features = torch.tensor(np.load(os.path.join(self.data_path, "dw_emb_features.npy")))
        node_type_mask = torch.tensor([int(node_type_mapping[r[0][0]]) for r in data_content['node2id']])
        node_id_node_label = torch.tensor([[node_id_mapping["p" + str(x[0])], x[1]] for x in data_content["paper_label"]]).T
        data_list = [Data(node_features=node_features,
                          node_type_mask=node_type_mask,
                          edge_index_dict=edge_index_dict,
                          node_id_node_label=node_id_node_label)]

        self.wrap_processing(data_list)


class DBLP_N_A(N_Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DBLP_N_A, self).__init__(root, "DBLP_N_A", transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data_content = self.read_data_files()
        node_type_mapping = {"a": "0", "p": "1", "c": "2"}
        node_id_mapping = {x[0]: int(x[1]) for x in data_content["node2id"]}

        relations = data_content['relations']
        edge_index_dict = {
            ('0', '1'): torch.from_numpy(relations[relations[:, 2] == 0][:, [0, 1]].T),
            ('1', '0'): torch.from_numpy(relations[relations[:, 2] == 0][:, [1, 0]].T),
            ('1', '2'): torch.from_numpy(relations[relations[:, 2] == 1][:, [0, 1]].T),
            ('2', '1'): torch.from_numpy(relations[relations[:, 2] == 1][:, [1, 0]].T)
        }
        node_features = torch.tensor(np.load(os.path.join(self.data_path, "dw_emb_features.npy")))
        node_type_mask = torch.tensor([int(node_type_mapping[r[0][0]]) for r in data_content['node2id']])
        node_id_node_label = torch.tensor([[node_id_mapping["a" + str(x[0])], x[1]] for x in data_content["author_label"]]).T
        data_list = [Data(node_features=node_features,
                          node_type_mask=node_type_mask,
                          edge_index_dict=edge_index_dict,
                          node_id_node_label=node_id_node_label)]

        self.wrap_processing(data_list)


class ACM_N(N_Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ACM_N, self).__init__(root, "ACM_N", transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data_content = self.read_data_files()
        node_type_mapping = {"a": "1", "p": "0", "s": "2"}
        node_id_mapping = {x[0]: int(x[1]) for x in data_content["node2id"]}

        relations = data_content['relations']
        edge_index_dict = {
            ('0', '1'): torch.from_numpy(relations[relations[:, 2] == 0][:, [0, 1]].T),
            ('1', '0'): torch.from_numpy(relations[relations[:, 2] == 0][:, [1, 0]].T),
            ('0', '2'): torch.from_numpy(relations[relations[:, 2] == 1][:, [0, 1]].T),
            ('2', '0'): torch.from_numpy(relations[relations[:, 2] == 1][:, [1, 0]].T)
        }
        node_features = torch.tensor(np.load(os.path.join(self.data_path, "dw_emb_features.npy")))
        node_type_mask = torch.tensor([int(node_type_mapping[r[0][0]]) for r in data_content['node2id']])
        node_id_node_label = torch.tensor([[node_id_mapping["p" + str(x[0])], x[1]] for x in data_content["p_label"]]).T
        data_list = [Data(node_features=node_features,
                          node_type_mask=node_type_mask,
                          edge_index_dict=edge_index_dict,
                          node_id_node_label=node_id_node_label)]

        self.wrap_processing(data_list)


class IMDB_N(N_Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(IMDB_N, self).__init__(root, "IMDB_N", transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data_content = self.read_data_files()

        node_type_mapping = {"m": "0", "a": "1", "d": "2"}
        node_id_mapping = {x[0]: int(x[1]) for x in data_content["node2id"]}

        relations = data_content['relations']
        edge_index_dict = {
            ('0', '1'): torch.from_numpy(relations[relations[:, 2] == 0][:, [0, 1]].T),
            ('1', '0'): torch.from_numpy(relations[relations[:, 2] == 0][:, [1, 0]].T),
            ('0', '2'): torch.from_numpy(relations[relations[:, 2] == 1][:, [0, 1]].T),
            ('2', '0'): torch.from_numpy(relations[relations[:, 2] == 1][:, [1, 0]].T)
        }
        node_features = torch.tensor(np.load(os.path.join(self.data_path, "dw_emb_features.npy")))
        node_type_mask = torch.tensor([int(node_type_mapping[r[0][0]]) for r in data_content['node2id']])
        node_id_node_label = torch.tensor((list(range(len(data_content["m_label"]))), data_content["m_label"]))
        data_list = [Data(node_features=node_features,
                          node_type_mask=node_type_mask,
                          edge_index_dict=edge_index_dict,
                          node_id_node_label=node_id_node_label)]
        self.wrap_processing(data_list)


