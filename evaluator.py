import torch
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

from code.extras import map_labels, predict_node_classification, Scores, kv_to_print_str


class Evaluator:
    def __init__(self, node_id_node_label, id_type_mask, target_ids):
        self.eval_ground_items = {}
        self.eval_results = {}
        for i in target_ids:
            ind_y_dict = {e[0]: e[1] for e in node_id_node_label.T.tolist() if e[0] in torch.where(id_type_mask == i)[0]}
            ind_emb = list(ind_y_dict.keys())
            y_emb = torch.tensor([ind_y_dict[k] for k in ind_emb])
            K = torch.unique(y_emb).shape[0]
            ind_train, ind_test, y_train, y_test = train_test_split(ind_emb, y_emb.numpy(), test_size=0.2, random_state=0)
            y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)
            self.eval_ground_items[i] = {
                "ind_y_dict": ind_y_dict,
                "ind_emb": ind_emb,
                "y_emb": y_emb,
                "K": K,
                "ind_train": ind_train,
                "ind_test": ind_test,
                "y_train": y_train,
                "y_test": y_test
            }

    def evaluate(self, z, y_pred_pc, y_pred_qc):
        for i in range(len(self.eval_ground_items.keys())):
            kmeans = KMeans(n_clusters=self.eval_ground_items[i]["K"], random_state=0, n_init=30).fit(z.detach().cpu().numpy())
            y_pred_kmeans = kmeans.labels_[self.eval_ground_items[i]["ind_emb"]]
            y_true = self.eval_ground_items[i]["y_emb"].numpy()
            y_pred_kmeans = map_labels(y_pred_kmeans, y_true)[1][y_pred_kmeans]
            y_pred_pc_i = y_pred_pc[self.eval_ground_items[i]["ind_emb"]]
            y_pred_qc_i = y_pred_qc[self.eval_ground_items[i]["ind_emb"]]
            y_pred_pc_i = map_labels(y_pred_pc_i, y_true)[1][y_pred_pc_i]
            y_pred_qc_i = map_labels(y_pred_qc_i, y_true)[1][y_pred_qc_i]
            ###### classification stuff start ####
            cls_pred = predict_node_classification(
                z[self.eval_ground_items[i]["ind_train"]],
                self.eval_ground_items[i]["y_train"],
                z[self.eval_ground_items[i]["ind_test"]])
            cls_f1_micro = f1_score(self.eval_ground_items[i]["y_test"], cls_pred, average='micro')
            cls_f1_macro = f1_score(self.eval_ground_items[i]["y_test"], cls_pred, average='macro')
            ###### classification stuff end ####

            nmi_kmeans = normalized_mutual_info_score(labels_true=y_true, labels_pred=y_pred_kmeans, average_method="arithmetic")
            nmi_pc = normalized_mutual_info_score(labels_true=y_true, labels_pred=y_pred_pc_i, average_method="arithmetic")
            nmi_qc = normalized_mutual_info_score(labels_true=y_true, labels_pred=y_pred_qc_i, average_method="arithmetic")
            # writer.add_embedding(z[ind_emb], metadata_header=["true", "pred_km", "pred_psi"],  global_step=epoch, metadata=torch.cat((
            #     torch.from_numpy(y_true).int()[:, None],
            #     torch.from_numpy(y_pred_kmeans).int()[:, None],
            #     torch.from_numpy(y_pred_psi).int()[:, None]), dim=1).detach().cpu().tolist())
            self.eval_results[i] = {
                Scores.NMI_KM: nmi_kmeans,
                Scores.NMI_Pc: nmi_pc,
                Scores.NMI_Qc: nmi_qc,
                Scores.F1_MICRO: cls_f1_micro,
                Scores.F1_MACRO: cls_f1_macro
            }

    def get_results_string(self):
        return "\t||\t".join([f'i={i:02d}\t||\t' + kv_to_print_str(self.eval_results[i], remove_str="Scores.")
                              for i in range(len(self.eval_ground_items.keys()))])
