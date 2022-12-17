import os
from itertools import groupby

import dgl
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, f1_score
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from code.datasets_loader import DBLP_N_P, DBLP_N_A
from code.evaluator import Evaluator
from code.extras import *
from code.model import Model




generate_new_mp_samples = True
z_dim = 24
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = DBLP_N_A(root=os.path.join(os.path.dirname(__file__), "..", "data"))
edge_index, edge_type = edge_type_and_index(ds[0])
node_feature_matrix = ds[0]["node_features"]
id_type_mask = ds[0]["node_type_mask"]
node_feature_matrix, edge_index = node_feature_matrix.to(device), edge_index.to(device)
N = id_type_mask.size()[0]
R = len(set(edge_type.tolist()))

ind_y_dict = {e[0]: e[1] for e in ds[0]["node_id_node_label"].T.tolist() if e[0] in torch.where(id_type_mask == 0)[0]}
ind_emb = list(ind_y_dict.keys())
y_emb = torch.tensor([ind_y_dict[k] for k in ind_emb])
K = torch.unique(y_emb).shape[0]
# ind_train, ind_test, y_train, y_test = train_test_split(ind_emb, y_emb.numpy(), stratify=y_emb, test_size=0.2, random_state=0)
ind_train, ind_test, y_train, y_test = train_test_split(ind_emb, y_emb.numpy(), test_size=0.2, random_state=0)
y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)

if generate_new_mp_samples:
    samples = generate_samples_new(ds[0].edge_index_dict, metapaths=["01210", "101"],
                               nodes_per_metapath=[10, 5, 5], save_name="samples.pkl")
    reshaped_samples = {
        ntype: [samples[ntype][k]['samples'].reshape(-1, samples[ntype][k]['N'], samples[ntype][k]['samples'].shape[1])
                for k in samples[ntype]] for ntype in samples}
    samples = [samples, reshaped_samples]
else:
    print("using old samples")
    with open("samples.pkl", "rb") as f:
        samples = pickle.load(f)

model = Model(N, R, K, node_feature_matrix.size(1), z_dim, device, samples)
writer = SummaryWriter()
node_feature_matrix = node_feature_matrix.float()
epoch_numbers = [50, 150]

#########################
evaluator = Evaluator(ds[0]["node_id_node_label"], id_type_mask, target_ids=[0])

@torch.no_grad()
def analyse():
    model.encoder.eval()
    model.decoder.eval()
    z, extras = model.encoder(node_feature_matrix, edge_index, edge_type)
    log_pc_given_z, log_qc_given_zA, psi, c = model.pc_and_qc_dists(z, edge_index, samples)
    y_pred_pc = log_pc_given_z.probs.argmax(dim=1).cpu().numpy()
    y_pred_qc = log_qc_given_zA.probs.argmax(dim=1).cpu().numpy()
    evaluator.evaluate(z, y_pred_pc=y_pred_pc, y_pred_qc=y_pred_qc)


@torch.no_grad()
def initialize_pretrain_vars(*args):
    model.encoder.eval()
    z, _ = model.encoder(*args)
    z = z.detach()
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=30).fit(z.cpu().numpy())
    logits = torch.log_softmax(-torch.from_numpy(kmeans.transform(z.cpu().numpy())) ** 1, dim=1).to(device)
    return z, logits

z = None
for epoch in range(1, 10001):
    if epoch < epoch_numbers[0]:
        z, losses = model.encoder_training_loop(node_feature_matrix, edge_index, edge_type, samples)
        if epoch == epoch_numbers[0] - 1:
            z, logits = initialize_pretrain_vars(node_feature_matrix, edge_index, edge_type)
    elif epoch_numbers[0] <= epoch < epoch_numbers[1]:
        f_args = node_feature_matrix, z, edge_index, edge_type
        extras, losses = model.decoder_training_loop(node_feature_matrix, z, edge_index, edge_type, logits, samples)
    else:
        extras, losses = model.joint_training_loop(node_feature_matrix, edge_index, edge_type, samples)

    if epoch > epoch_numbers[1]:
        analyse()
        print(f'Epoch: {epoch:02d}\t' + kv_to_print_str(losses, remove_str="Losses.") + "\t||\t" + evaluator.get_results_string())

print("done")
