import opt
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Module, Parameter

import numpy as np
from scipy.stats import qmc


class AE_encoder(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)
        self.z_layer = Linear(ae_n_enc_3, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z = self.act(self.enc_3(z))
        z_ae = self.z_layer(z)
        return z_ae


class AE_decoder(nn.Module):
    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE_decoder, self).__init__()

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        z = self.act(self.dec_3(z))
        x_hat = self.x_bar_layer(z)
        return x_hat


class AE(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_input=n_input,
            n_z=n_z)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if opt.args.name == "dblp" or opt.args.name == "hhar":
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        elif opt.args.name == "reut":
            self.act = nn.LeakyReLU(0.2, inplace=True)
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            if opt.args.name == "dblp" or opt.args.name == "hhar":
                support = self.act(F.linear(features, self.weight))
            else:
                support = self.act(torch.mm(features, self.weight))
        else:
            if opt.args.name == "dblp" or opt.args.name == "hhar":
                support = F.linear(features, self.weight)
            else:
                support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        az = torch.spmm(adj, output)
        return output, az


class IGAE_encoder(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z_1, az_1 = self.gnn_1(x, adj, active=False if opt.args.name == "hhar" else True)
        z_2, az_2 = self.gnn_2(z_1, adj, active=False if opt.args.name == "hhar" else True)
        z_igae, az_3 = self.gnn_3(z_2, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj, [az_1, az_2, az_3], [z_1, z_2, z_igae]


class IGAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z_1, az_1 = self.gnn_4(z_igae, adj, active=False if opt.args.name == "hhar" else True)
        z_2, az_2 = self.gnn_5(z_1, adj, active=False if opt.args.name == "hhar" else True)
        z_hat, az_3 = self.gnn_6(z_2, adj, active=False if opt.args.name == "hhar" else True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj, [az_1, az_2, az_3], [z_1, z_2, z_hat]

class IGAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)


# Relational Redundancy-Free Graph Clustering
class R2FGC(nn.Module):
    def __init__(self, diffusion=None, degree=None, not_zero=None, device=None, n_node=None):
        super(R2FGC, self).__init__()

        # AE
        self.ae = AE(
            ae_n_enc_1=opt.args.ae_n_enc_1,
            ae_n_enc_2=opt.args.ae_n_enc_2,
            ae_n_enc_3=opt.args.ae_n_enc_3,
            ae_n_dec_1=opt.args.ae_n_dec_1,
            ae_n_dec_2=opt.args.ae_n_dec_2,
            ae_n_dec_3=opt.args.ae_n_dec_3,
            n_input=opt.args.n_input,
            n_z=opt.args.n_z)

        # GAE
        self.gae = IGAE(
            gae_n_enc_1=opt.args.gae_n_enc_1,
            gae_n_enc_2=opt.args.gae_n_enc_2,
            gae_n_enc_3=opt.args.gae_n_enc_3,
            gae_n_dec_1=opt.args.gae_n_dec_1,
            gae_n_dec_2=opt.args.gae_n_dec_2,
            gae_n_dec_3=opt.args.gae_n_dec_3,
            n_input=opt.args.n_input)

        # fusion parameter
        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5), requires_grad=True)
        self.alpha = Parameter(torch.zeros(1))

        # constrastive
        # global
        self.topk = opt.args.sample
        # local 
        self.diffusion = diffusion

        self.degree = degree / degree.sum()
        self.not_zero = not_zero
        self.device = device

        # cluster layer (clustering assignment matrix)
        self.cluster_centers = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)

    # calculate the soft assignment distribution Q
    def q_distribute(self, Z, Z_ae, Z_igae):
        """
        calculate the soft assignment distribution based on the embedding and the cluster centers
        Args:
            Z: fusion node embedding
            Z_ae: node embedding encoded by AE
            Z_igae: node embedding encoded by GAE
        Returns:
            the soft assignment distribution Q
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_centers, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(Z_ae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_igae = 1.0 / (1.0 + torch.sum(torch.pow(Z_igae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_igae = (q_igae.t() / torch.sum(q_igae, 1)).t()

        return [q, q_ae, q_igae]

    def forward(self, Ad, Am, x1, x2, device=None):

        Ad = Ad.to(device)
        Am = Am.to(device)

        # node embedding encoded by AE
        Z_ae1 = self.ae.encoder(x1)
        Z_ae2 = self.ae.encoder(x2)

        # node embedding encoded by GAE
        Z_igae1, A_igae1, AZ_1, Z_1 = self.gae.encoder(x1, Am)
        Z_igae2, A_igae2, AZ_2, Z_2 = self.gae.encoder(x2, Ad)

        # AE
        pred1 = F.normalize(Z_ae1, p=2.0, dim=1)
        pred2 = F.normalize(Z_ae2, p=2.0, dim=1)

        with torch.no_grad():
            teacher1 = F.normalize(Z_ae1, p=2.0, dim=1)
            teacher2 = F.normalize(Z_ae2, p=2.0, dim=1)
            
            # sample = torch.tensor(np.random.choice(x1.shape[0], size = (self.topk), replace=False, p=self.degree)).to(self.device)
            # sample = torch.tensor(np.random.choice(x1.shape[0], size = (self.topk), replace=True)).to(self.device)
            # engine = qmc.MultinomialQMC(pvals=self.degree)
            # data_idx = np.array(range(x1.shape[0]))
            # sample_fre = engine.random(self.topk)
            # sample = np.repeat(data_idx, sample_fre)

            engine = qmc.MultinomialQMC(pvals=self.degree, n_trials=self.topk)
            sample_fre = engine.random(1).astype(int)
            data_idx = np.array(range(x1.shape[0]))
            sample = data_idx[np.nonzero(sample_fre[0])[0]]
                    
        sim_q1_ae = torch.mm(pred1, teacher2[sample].T)
        sim_k1_ae = torch.mm(teacher2, teacher2[sample].T)

        sim_q2_ae = torch.mm(pred2, teacher1[sample].T)
        sim_k2_ae = torch.mm(teacher1, teacher1[sample].T)

        sim_q1_diffusion_ae = torch.bmm(teacher2[self.diffusion], pred1.reshape(pred1.shape[0], pred1.shape[1], 1)).squeeze(2)
        sim_k1_diffusion_ae = torch.bmm(teacher2[self.diffusion], teacher2.reshape(teacher2.shape[0], teacher2.shape[1], 1)).squeeze(2)

        sim_q2_diffusion_ae = torch.bmm(teacher1[self.diffusion], pred2.reshape(pred2.shape[0], pred2.shape[1], 1)).squeeze(2)
        sim_k2_diffusion_ae = torch.bmm(teacher1[self.diffusion], teacher1.reshape(teacher1.shape[0], teacher1.shape[1], 1)).squeeze(2)


        # GAE
        pred1 = F.normalize(Z_igae1, p=2.0, dim=1)
        pred2 = F.normalize(Z_igae2, p=2.0, dim=1)

        with torch.no_grad():
            teacher1 = F.normalize(Z_igae1, p=2.0, dim=1)
            teacher2 = F.normalize(Z_igae2, p=2.0, dim=1)

            # sample = torch.tensor(np.random.choice(x1.shape[0], size = (self.topk), replace=False, p=self.degree)).to(self.device)
            # sample = torch.tensor(np.random.choice(x1.shape[0], size = (self.topk), replace=True)).to(self.device)
            # engine = qmc.MultinomialQMC(pvals=self.degree)
            # data_idx = np.array(range(x1.shape[0]))
            # sample_fre = engine.random(self.topk)
            # sample = np.repeat(data_idx, sample_fre)

            engine = qmc.MultinomialQMC(pvals=self.degree, n_trials=self.topk)
            sample_fre = engine.random(1).astype(int)
            data_idx = np.array(range(x1.shape[0]))
            sample = data_idx[np.nonzero(sample_fre[0])[0]]

        sim_q1_gae = torch.mm(pred1, teacher2[sample].T)
        sim_k1_gae = torch.mm(teacher2, teacher2[sample].T)

        sim_q2_gae = torch.mm(pred2, teacher1[sample].T)
        sim_k2_gae = torch.mm(teacher1, teacher1[sample].T)

        sim_q1_diffusion_gae = torch.bmm(teacher2[self.diffusion], pred1.reshape(pred1.shape[0], pred1.shape[1], 1)).squeeze(2)
        sim_k1_diffusion_gae = torch.bmm(teacher2[self.diffusion], teacher2.reshape(teacher2.shape[0], teacher2.shape[1], 1)).squeeze(2)

        sim_q2_diffusion_gae = torch.bmm(teacher1[self.diffusion], pred2.reshape(pred2.shape[0], pred2.shape[1], 1)).squeeze(2)
        sim_k2_diffusion_gae = torch.bmm(teacher1[self.diffusion], teacher1.reshape(teacher1.shape[0], teacher1.shape[1], 1)).squeeze(2)

        Z_ae = (Z_ae1 + Z_ae2) / 2
        Z_igae = (Z_igae1 + Z_igae2) / 2

        Z_i = self.a * Z_ae + self.b * Z_igae
        Z_l = torch.spmm(Am, Z_i)
        S = torch.mm(Z_l, Z_l.t())
        S = F.softmax(S, dim=1)
        Z_g = torch.mm(S, Z_l)
        Z = self.alpha * Z_g + Z_l

        # AE decoding
        X_hat = self.ae.decoder(Z)

        # GAE decoding
        Z_hat, Z_adj_hat, AZ_de, Z_de = self.gae.decoder(Z, Am)
        sim = (A_igae1 + A_igae2) / 2
        A_hat = sim + Z_adj_hat

        # ae and gae embedding
        Z_ae_all = [Z_ae1, Z_ae2]
        Z_gae_all = [Z_igae1, Z_igae2]

        # the soft assignment distribution Q
        Q = self.q_distribute(Z, Z_ae, Z_igae)

        # propagated embedding AZ_all and embedding Z_all
        AZ_en = []
        Z_en = []
        for i in range(len(AZ_1)):
            AZ_en.append((AZ_1[i]+AZ_2[i])/2)
            Z_en.append((Z_1[i]+Z_2[i])/2)
        AZ_all = [AZ_en, AZ_de]
        Z_all = [Z_en, Z_de]

        return X_hat, Z_hat, A_hat, sim, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all, \
            [sim_q1_ae, sim_q1_diffusion_ae[self.not_zero], sim_q1_gae, sim_q1_diffusion_gae[self.not_zero]], \
                [sim_k1_ae, sim_k1_diffusion_ae[self.not_zero], sim_k1_gae, sim_k1_diffusion_gae[self.not_zero]], \
                    [sim_q2_ae, sim_q2_diffusion_ae[self.not_zero], sim_q2_gae, sim_q2_diffusion_gae[self.not_zero]], \
                        [sim_k2_ae, sim_k2_diffusion_ae[self.not_zero], sim_k2_gae, sim_k2_diffusion_gae[self.not_zero]]
