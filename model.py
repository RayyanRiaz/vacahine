import itertools

from torch.distributions import kl_divergence, Categorical
from torch_scatter import scatter_mean

from code.extras import kl_categorical_categorical
from code.modules import *
import torch

from code.extras import get_negative_edge_index, Losses as L


class PsiMatrixNSHE2(nn.Module):
    def __init__(self, N, R, K, z_dim):
        super(PsiMatrixNSHE2, self).__init__()
        self.psi = nn.Parameter(torch.randn(K, z_dim), requires_grad=True)
        self.psi_mp = nn.Parameter(torch.randn(z_dim, z_dim), requires_grad=True)
        uniform(K, self.psi)
        uniform(z_dim, self.psi)

    def forward(self, *args):
        return self.psi


class InnerProductDecoderNSHE2(nn.Module):
    def __init__(self):
        super(InnerProductDecoderNSHE2, self).__init__()

    def forward(self, z, edge_index, c, psi, mp_samples):
        row, col = edge_index
        if c is not None:
            assert psi is not None
            assert c.shape[0] == z.shape[0]
            psi_c = (c[:, :, None] * psi[None, :, :]).sum(1)
            return ((z[row] * psi_c[col]).sum(dim=1).sigmoid() + (z[col] * psi_c[row]).sum(dim=1).sigmoid()) / 2
        else:
            return (z[row] * z[col]).sum(dim=1).sigmoid()

    def loss(self, z, pos_edge_index, neg_edge_index, c=None, psi=None, mp_samples=None):
        pos_loss = -torch.log(self.forward(z, pos_edge_index, c, psi, mp_samples) + EPS).mean()
        neg_loss = -torch.log(1 - self.forward(z, neg_edge_index, c, psi, mp_samples) + EPS).mean()
        return pos_loss + neg_loss, pos_loss, neg_loss


class Model:
    def __init__(self, N, R, K, feature_dim, z_dim, device, samples):
        self.encoder = RGCN_V_Encoder(feature_dim, z_dim, R).to(device)
        self.decoder = InnerProductDecoderNSHE2().to(device)
        self.discriminator = DGIDiscriminator(z_dim * 5, z_dim).to(device)
        self.psi_model = PsiMatrixNSHE2(N, R, K, z_dim).to(device)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=0.01, weight_decay=0.0005)
        self.psi_opt = torch.optim.Adam(itertools.chain(self.psi_model.parameters(), self.discriminator.parameters()), lr=0.01)

        self.discriminators = {k: {mp: DGIDiscriminator(z_dim, z_dim).to(device) for mp in samples[0][k]} for k in samples[0]}
        self.discriminators_opt = torch.optim.Adam(
            itertools.chain(*[self.discriminators[ntype][mp].parameters() for ntype in self.discriminators for mp in self.discriminators[ntype]]),
            lr=0.01)


    def pc_and_qc_dists(self, z, edge_index, mp_samples, q_alpha=0.9):
        row, col = edge_index
        psi = self.psi_model()
        dot_products = (psi[None, :, :] * z[:, None, :]).sum(dim=2)
        dot_products_avg_over_Ni = scatter_mean(src=dot_products[row], index=col, dim=0, dim_size=z.size(0))
        weighted_dot_products = q_alpha * dot_products + (1 - q_alpha) * dot_products_avg_over_Ni
        z_means = [torch.cat([z[samples].mean(dim=(1, 2))[None, :, :] for samples in mp_samples[1][ntype]], dim=0).sum(dim=0)
                   for ntype in mp_samples[1]]
        weighted_dot_products[0:z_means[0].shape[0]] = (psi[None, :, :] * z_means[0][:, None, :]).sum(dim=2)
        weighted_dot_products[z_means[0].shape[0]:z_means[0].shape[0] + z_means[1].shape[0]] = (psi[None, :, :] * z_means[1][:, None, :]).sum(dim=2)
        c = F.gumbel_softmax(logits=weighted_dot_products, tau=1, hard=True)
        return Categorical(logits=dot_products), Categorical(logits=weighted_dot_products), psi, c

    def corrupt(self, feature_matrix, samples_matrix):
        feature_matrix_corr = feature_matrix[torch.randperm(feature_matrix.size(0))]
        rand_perm_idx = torch.randint(1, 4, (1,))[0]
        return feature_matrix_corr, samples_matrix

    def loss_dgi_subgraphs(self, x, z, edge_index, edge_type, samples):
        loss = 0
        for ntype in samples[0]:
            for mp in samples[0][ntype]:
                pos_samples = samples[0][ntype][mp]['samples']
                z_mp = z[pos_samples]
                summaries = z_mp.mean(dim=1)
                z_mp_flattened = z_mp.reshape(pos_samples.shape[0], -1)
                pos_decisions = self.discriminators[ntype][mp](z_mp, summaries)

                node_feature_matrix_corr, samples_corr = self.corrupt(x, pos_samples)
                z_neg, extas_n = self.encoder(node_feature_matrix_corr, edge_index, edge_type)
                z_neg_mp = z_neg[samples_corr]
                z_neg_mp_flattened = z_neg_mp.reshape(samples_corr.shape[0], -1)
                neg_decisions = self.discriminators[ntype][mp](z_neg_mp, summaries)
                loss += (-torch.log(pos_decisions + EPS).mean() - torch.log(1 - neg_decisions + EPS).mean())
        return loss

    def encoder_training_loop(self, x, edge_index, edge_type, mp_samples):
        self.encoder.train()
        self.encoder_opt.zero_grad()
        z, extras = self.encoder(x, edge_index, edge_type)
        enc_loss = self.encoder.loss(*extras)
        neg_edge_index = get_negative_edge_index(edge_index, z.size(0))
        recon_loss, _, _ = self.decoder.loss(z, edge_index, neg_edge_index)
        l_dgi_subgraphs = 0
        loss = recon_loss + enc_loss + l_dgi_subgraphs
        loss.backward()
        self.encoder_opt.step()
        return z, {L.L_Total: loss, L.L_Recon: recon_loss, L.L_Enc: enc_loss}

    def decoder_training_loop(self, x, z, edge_index, edge_type, logits, mp_samples):
        self.encoder.train()
        self.decoder.train()
        self.encoder_opt.zero_grad()
        self.psi_opt.zero_grad()
        self.discriminators_opt.zero_grad()
        z, extras = self.encoder(x, edge_index, edge_type)
        enc_loss = self.encoder.loss(*extras)
        log_pc_given_z, log_qc_given_zA, psi, c = self.pc_and_qc_dists(z, edge_index, mp_samples)
        l_kmeans_logits = kl_categorical_categorical(log_pc_given_z, Categorical(logits=logits)).mean()
        neg_edge_index = get_negative_edge_index(edge_index, z.size(0))
        recon_loss, _, _ = self.decoder.loss(z, edge_index, neg_edge_index, c, psi, mp_samples)
        l_dgi_subgraphs = self.loss_dgi_subgraphs(x, z, edge_index, edge_type, mp_samples)
        loss = recon_loss + l_kmeans_logits + enc_loss + 1*l_dgi_subgraphs
        loss.backward()
        self.encoder_opt.step()
        self.psi_opt.step()
        self.discriminators_opt.step()
        return (z, x, psi, c, neg_edge_index), \
               {L.L_Total: loss, L.L_Enc: enc_loss, L.L_Recon: recon_loss, L.L_KMeans_logits: l_kmeans_logits, L.L_DGI_subgraphs: l_dgi_subgraphs}

    def joint_training_loop(self, x, edge_index, edge_type, mp_samples):
        self.encoder.train()
        self.decoder.train()
        self.encoder_opt.zero_grad()
        self.psi_opt.zero_grad()
        self.discriminators_opt.zero_grad()

        z, extras = self.encoder(x, edge_index, edge_type)
        enc_loss = self.encoder.loss(*extras)
        log_pc_given_z, log_qc_given_zA, psi, c = self.pc_and_qc_dists(z, edge_index, mp_samples)
        dists_loss = kl_categorical_categorical(log_pc_given_z, log_qc_given_zA).mean()
        neg_edge_index = get_negative_edge_index(edge_index, z.size(0))
        recon_loss, _, _ = self.decoder.loss(z, edge_index, neg_edge_index, c, psi, mp_samples)
        l_dgi_subgraphs = self.loss_dgi_subgraphs(x, z, edge_index, edge_type, mp_samples)

        loss = recon_loss + dists_loss + enc_loss + 1*l_dgi_subgraphs
        loss.backward()
        self.encoder_opt.step()
        self.psi_opt.step()
        self.discriminators_opt.step()

        return (z, c), {L.L_Total: loss, L.L_Recon: recon_loss, L.L_KL_p_and_qc: dists_loss, L.L_Enc: enc_loss, L.L_DGI_subgraphs: l_dgi_subgraphs}

