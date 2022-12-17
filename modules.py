import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GCNConv
from torch_geometric.nn.inits import uniform

from extras import EPS, MAX_LOGSTD


class Encoder(nn.Module):

    def loss(self, *args):
        return torch.tensor(0.)


class V_Encoder(Encoder):

    def reparametrize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            return mu

    def loss(self, mu=None, logstd=None):
        logstd = logstd.clamp(max=MAX_LOGSTD)
        loss = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))
        loss /= mu.shape[0]
        return loss


class GCNEncoder(Encoder):
    def __init__(self, in_dim, z_dim):
        super(GCNEncoder, self).__init__()
        self.conv = GCNConv(in_dim, z_dim, cached=True)
        self.prelu = nn.PReLU(z_dim)

    def forward(self, x, edge_index, edge_type):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x, ()


class GCN_V_Encoder(V_Encoder):
    def __init__(self, in_dim, z_dim, R):
        super(GCN_V_Encoder, self).__init__()
        self.conv1 = GCNConv(in_dim, z_dim * 2, cached=True)
        self.conv_mu = GCNConv(z_dim * 2, z_dim, cached=True)
        self.conv_logstd = GCNConv(z_dim * 2, z_dim, cached=True)

    def forward(self, x, edge_index, edge_type):
        z = self.conv1(x, edge_index)
        z = F.leaky_relu(z)
        mu = self.conv_mu(z, edge_index)
        logstd = self.conv_logstd(z, edge_index)
        z = self.reparametrize(mu, logstd)
        return (z, (mu, logstd)) if self.training else (mu, ())


class RGCNEncoder(Encoder):
    def __init__(self, in_dim, z_dim, R):
        super(RGCNEncoder, self).__init__()
        self.conv1 = RGCNConv(in_dim, z_dim * 2, R, num_bases=30)
        self.conv2 = RGCNConv(z_dim * 2, z_dim, R, num_bases=30)

    def forward(self, x, edge_index, edge_type):
        z = self.conv1(x, edge_index, edge_type)
        z = F.leaky_relu(z)
        z = self.conv2(z, edge_index, edge_type)
        return z, ()


class Matrices_V_Encoder(V_Encoder):
    def __init__(self, in_channels, out_channels):
        super(Matrices_V_Encoder, self).__init__()
        self.mu = nn.Parameter(torch.randn(in_channels, out_channels), requires_grad=True)
        self.logvar = nn.Parameter(torch.randn(in_channels, out_channels), requires_grad=True)

        uniform(out_channels, self.mu)
        uniform(out_channels, self.logvar)

    def forward(self, x, edge_index, *args):
        mu, logvar = x.matmul(self.mu), x.matmul(self.logvar)
        z = self.reparametrize(mu, logvar)
        return (z, (mu, logvar)) if self.training else (mu, ())


class RGCN_V_Encoder(V_Encoder):
    def __init__(self, in_dim, z_dim, R):
        super(RGCN_V_Encoder, self).__init__()
        self.conv1 = RGCNConv(in_dim, z_dim * 2, R, num_bases=30)
        self.conv_mu = RGCNConv(z_dim * 2, z_dim, R, num_bases=30)
        self.conv_logstd = RGCNConv(z_dim * 2, z_dim, R, num_bases=30)

    def forward(self, x, edge_index, edge_type):
        z = self.conv1(x, edge_index, edge_type)
        z = F.leaky_relu(z)
        mu = self.conv_mu(z, edge_index, edge_type)
        logstd = self.conv_logstd(z, edge_index, edge_type)
        z = self.reparametrize(mu, logstd)
        return (z, (mu, logstd)) if self.training else (mu, ())


class PsiMatrix(nn.Module):
    def __init__(self, N, R, K, z_dim):
        super(PsiMatrix, self).__init__()
        self.psi = nn.Parameter(torch.randn(K, z_dim), requires_grad=True)
        uniform(K, self.psi)

    def forward(self, *args):
        return self.psi


class DGIDiscriminator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DGIDiscriminator, self).__init__()
        self.mat = nn.Parameter(torch.Tensor(in_dim, out_dim))
        uniform(in_dim, self.mat)

    def forward(self, z, s):
        return ((z @ self.mat) * s[:, None, :]).sum(dim=2).sigmoid()


class FCNet(nn.Module):
    def __init__(self, sizes, last_layer_activation):
        super(FCNet, self).__init__()

        net = []
        for i in range(1, len(sizes)):
            net.append(nn.Linear(sizes[i - 1], sizes[i]))
            if i == len(sizes) - 1:
                if last_layer_activation is not None:
                    net.append(last_layer_activation)
            else:

                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class LinearSummary(nn.Module):
    def __init__(self, dims):
        super(LinearSummary, self).__init__()
        self.net = FCNet(dims, last_layer_activation=nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, z, edge_index, c, psi):
        row, col = edge_index
        if c is not None:
            assert psi is not None
            assert c.shape[0] == z.shape[0]

            psi_c = (c[:, :, None] * psi[None, :, :]).sum(1)
            return ((z[row] * psi_c[col]).sum(dim=1).sigmoid() + (z[col] * psi_c[row]).sum(dim=1).sigmoid()) / 2
        else:
            return (z[row] * z[col]).sum(dim=1).sigmoid()

    def loss(self, z, pos_edge_index, neg_edge_index, c=None, psi=None):
        pos_loss = -torch.log(self.forward(z, pos_edge_index, c, psi) + EPS).mean()
        neg_loss = -torch.log(1 - self.forward(z, neg_edge_index, c, psi) + EPS).mean()
        return pos_loss + neg_loss, pos_loss, neg_loss
