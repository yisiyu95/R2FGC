import numpy as np
from utils import *
from torch.optim import Adam

import scipy.sparse as sp
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data

from scipy.stats import qmc
import time

def train(model, data, X, y, A_norm, A, device):
    """
    train our model
    Args:
        model: Relational Redundancy-Free Graph Clustering
        X: input feature matrix
        y: input label
        A: input origin adj
        A_norm: normalized adj
    Returns: acc, nmi, ari, f1
    """
    print("Training…")
    # calculate embedding similarity and cluster centers
    sim, centers = model_init(model, data, y, device)
    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)
    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    Ad = diffusion_adj(A, mode="ppr", transport_rate=opt.args.eta_value).astype(np.float32)
    Am = remove_edge(A, sim, remove_rate=0.1).astype(np.float32)
    Ad = numpy_to_torch(Ad)
    Am = numpy_to_torch(Am)

    tmp_coo1 = sp.coo_matrix(Ad) 
    edge_index1, edge_attr1 = tg_utils.from_scipy_sparse_matrix(tmp_coo1)
    tmp_coo2 = sp.coo_matrix(Am) 
    edge_index2, edge_attr2 = tg_utils.from_scipy_sparse_matrix(tmp_coo2)

    starttime = time.time()
    for epoch in range(opt.args.epochs):

        X_tilde1, X_tilde2 = gaussian_noised_feature(X.to(device))
        view1 = Data(x=X_tilde1, edge_index=edge_index1, edge_attr=edge_attr1).to(device)
        view2 = Data(x=X_tilde2, edge_index=edge_index2, edge_attr=edge_attr2).to(device)

        X_hat, Z_hat, A_hat, _, _, _, Q, Z, AZ_all, Z_all, sim_q1, sim_k1, sim_q2, sim_k2 = model(Ad, Am, view1.x, view2.x, device=device)

        loss_REC = reconstruction_loss(X.to(device), A_norm.to(device), X_hat, Z_hat, A_hat)
        loss_DEG_DIFF = deg_diff_loss(sim_q1, sim_k1, sim_q2, sim_k2)
        loss_R = r_loss(AZ_all, Z_all)
        loss_KL = distribution_loss(Q, target_distribution(Q[0].data))
        loss_overall = loss_DEG_DIFF + loss_REC + opt.args.kappa_value * loss_KL + opt.args.epsilon_value * loss_R

        # optimization
        optimizer.zero_grad()
        loss_overall.backward()
        optimizer.step()
        res1 = Q[0].cpu().detach().numpy().argmax(1)
        acc, nmi, ari, f1 = eva(y, res1, epoch)

        if acc > opt.args.acc:
            opt.args.acc = acc
            opt.args.nmi = nmi
            opt.args.ari = ari
            opt.args.f1 = f1
            opt.args.best_epoch = epoch
            opt.args.best_Z = Z
            opt.args.prelabel = res1

        print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))
    
    endtime = time.time()
    run_time = round(endtime - starttime, 2)

    return opt.args.acc, opt.args.nmi, opt.args.ari, opt.args.f1, opt.args.best_epoch, opt.args.best_Z, opt.args.prelabel, run_time
