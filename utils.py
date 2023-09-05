import opt
import torch
import random
import numpy as np
from sklearn import metrics
from munkres import Munkres
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import torch_geometric.utils as tg_utils


def setup():
    """
    setup
    - name: the name of dataset
    - seed: random seed
    - n_clusters: num of cluster
    - n_input: dimension of feature
    - eta_value: eta value for graph diffusion
    - kappa_value: kappa value for clustering guidance
    - epsilon_value: epsilon value for propagation regularization
    - lr: learning rate
    - sample: number of global anchors
    - topk: number of local anchors
    - epochs: number of epochs
    Return: None

    """
    print("setting:")
    setup_seed(opt.args.seed)
    if opt.args.name == 'acm': # 3025 1870
        opt.args.n_clusters = 3
        opt.args.n_input = 100
        opt.args.eta_value = 0.2 # augmentation
        opt.args.kappa_value = 10 # KL loss_KL
        opt.args.epsilon_value = 5e3 # over-smoothing loss_R
        opt.args.lr = 5e-5
        opt.args.sample = 256
        opt.args.topk = 8
        opt.args.epochs = 600
    
    elif opt.args.name == 'amap': # 7650 745
        opt.args.n_clusters = 8
        opt.args.n_input = 100
        opt.args.eta_value = 0.2
        opt.args.kappa_value = 10
        opt.args.epsilon_value = 5e3
        opt.args.lr = 1e-3
        opt.args.sample = 256
        opt.args.topk = 8
        opt.args.epochs = 300

    elif opt.args.name == 'cite': # 3327 3703
        opt.args.n_clusters = 6
        opt.args.n_input = 100
        opt.args.eta_value = 0.2
        opt.args.kappa_value = 10
        opt.args.epsilon_value = 5e3
        opt.args.lr = 1e-3
        opt.args.sample = 256
        opt.args.topk = 6
        opt.args.epochs = 600

    elif opt.args.name == 'dblp': # 4057 334
        opt.args.n_clusters = 4
        opt.args.n_input = 50
        opt.args.eta_value = 0.2
        opt.args.kappa_value = 10
        opt.args.epsilon_value = 5e3
        opt.args.lr = 1e-4
        opt.args.sample = 256
        opt.args.topk = 128
        opt.args.epochs = 300
    
    elif opt.args.name == 'hhar': # 10299 561
        opt.args.n_clusters = 6
        opt.args.n_input = 50
        opt.args.eta_value = 0.2
        opt.args.kappa_value = 10
        opt.args.epsilon_value = 5e3
        opt.args.lr = 1e-3
        opt.args.sample = 256
        opt.args.topk = 8
        opt.args.epochs = 300

    else:
        print("error!")
        print("please add the new dataset's parameters")
        print("------------------------------")
        print("dataset       : ")
        print("device        : ")
        print("random seed   : ")
        print("clusters      : ")
        print("eta value   : ")
        print("kappa value  : ")
        print("epsilon value   : ")
        print("learning rate : ")
        print("------------------------------")
        exit(0)

    opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("random seed   : {}".format(opt.args.seed))
    print("clusters      : {}".format(opt.args.n_clusters))
    print("eta value     : {}".format(opt.args.eta_value))
    print("kappa value   : {}".format(opt.args.kappa_value))
    print("epsilon value : {:.0e}".format(opt.args.epsilon_value))
    print("learning rate : {:.0e}".format(opt.args.lr))
    print("------------------------------")


def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a


def load_graph_data(dataset_name, show_details=False):
    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :return: the features, labels and adj
    """

    load_path = "dataset/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")

    # X pre-processing
    pca = PCA(n_components=opt.args.n_input)
    feat = pca.fit_transform(feat)
    return feat, label, adj


def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)

    return norm_adj


def gaussian_noised_feature(X):
    """
    add gaussian noise to the attribute matrix X
    Args:
        X: the attribute matrix
    Returns: the noised attribute matrix X_tilde
    """
    N_1 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(opt.args.device)
    N_2 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(opt.args.device)
    X_tilde1 = X * N_1
    X_tilde2 = X * N_2
    return X_tilde1, X_tilde2


def diffusion_adj(adj, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # add the self_loop
    adj_tmp = adj + np.eye(adj.shape[0])

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)
    sqrt_d_inv = np.sqrt(d_inv)

    # calculate norm adj
    norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # calculate graph diffusion
    if mode == "ppr":
        diff_adj = transport_rate * np.linalg.inv((np.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))

    return diff_adj


def remove_edge(A, similarity, remove_rate=0.1):
    """
    remove edge based on embedding similarity
    Args:
        A: the origin adjacency matrix
        similarity: cosine similarity matrix of embedding
        remove_rate: the rate of removing linkage relation
    Returns:
        Am: edge-masked adjacency matrix
    """
    # remove edges based on cosine similarity of embedding
    n_node = A.shape[0]
    for i in range(n_node):
        A[i, torch.argsort(similarity[i].cpu())[:int(round(remove_rate * n_node))]] = 0

    # normalize adj
    Am = normalize_adj(A, self_loop=True, symmetry=False)
    return Am


def load_pretrain_parameter(model):
    """
    load pretrained parameters
    Args:
        model: Relational Redundancy-Free Graph Clustering
    Returns: model
    """
    pretrained_dict = torch.load('model_pretrain/{}_pretrain.pkl'.format(opt.args.name), map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def model_init(model, data, y, device):
    """
    load the pre-train model and calculate similarity and cluster centers
    Args:
        model: Relational Redundancy-Free Graph Clustering
        X: input feature matrix
        y: input label
        A_norm: normalized adj
    Returns: embedding similarity matrix
    """
    # load pre-train model
    model = load_pretrain_parameter(model)

    A = torch.Tensor(tg_utils.to_scipy_sparse_matrix(data.edge_index, data.edge_attr).todense()).to(device)

    # calculate embedding similarity
    with torch.no_grad():
        _, _, _, sim, _, _, _, Z, _, _, _, _, _, _ = model(A, A, x1=data.x, x2=data.x, device=device)

    # calculate cluster centers
    acc, nmi, ari, f1, centers = clustering(Z, y)

    return sim, centers


def reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat):
    """
    reconstruction loss L_{}
    Args:
        X: the origin feature matrix
        A_norm: the normalized adj
        X_hat: the reconstructed X
        Z_hat: the reconstructed Z
        A_hat: the reconstructed A
    Returns: the reconstruction loss
    """
    loss_ae = F.mse_loss(X_hat, X)
    loss_w = F.mse_loss(Z_hat, torch.spmm(A_norm, X))
    loss_a = F.mse_loss(A_hat, A_norm)
    loss_igae = loss_w + 0.1 * loss_a
    loss_rec = loss_ae + loss_igae
    return loss_rec


def target_distribution(Q):
    """
    calculate the target distribution (student-t distribution)
    Args:
        Q: the soft assignment distribution
    Returns: target distribution P
    """
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P


def distribution_loss(Q, P):
    """
    calculate the clustering guidance loss L_{KL}
    Args:
        Q: the soft assignment distribution
        P: the target distribution
    Returns: L_{KL}
    """
    loss = F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction='batchmean')
    return loss


def r_loss(AZ, Z):
    """
    the loss of propagated regularization (L_R)
    Args:
        AZ: the propagated embedding
        Z: embedding
    Returns: L_R
    """
    loss = 0
    for i in range(2):
        for j in range(3):
            p_output = F.softmax(AZ[i][j], dim=1)
            q_output = F.softmax(Z[i][j], dim=1)
            log_mean_output = ((p_output + q_output) / 2).log()
            loss += (F.kl_div(log_mean_output, p_output, reduction='batchmean') +
                     F.kl_div(log_mean_output, p_output, reduction='batchmean')) / 2
    return loss


def off_diagonal(x):
    """
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cross_correlation(Z_v1, Z_v2):
    """
    calculate the augmentation-based correlation matrix S
    Args:
        Z_v1: the first view embedding
        Z_v2: the second view embedding
    Returns: S
    """
    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())


def correlation_reduction_loss(S):
    """
    the correlation reduction loss L: MSE for S and I (identical matrix)
    Args:
        S: the augmentation-based correlation matrix S
    Returns: L
    """
    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()
    # return torch.diagonal(S).add(-1).pow(2).mean() # REpre
    # return off_diagonal(S).pow(2).mean() # REred



def deg_diff_loss(sim_q1, sim_k1, sim_q2, sim_k2):
    """
    Relation loss L_{RE}
    Args:
        sim_q1: [G_ae, L_ae, G_gae, L_ae] (student1 v.s. teacher2)
        sim_k1: constraction of sim_q1 (teacher2 v.s. teacher2)
        sim_q2: [G_ae, L_ae, G_gae, L_ae] (student2 v.s. teacher1)
        sim_k2: constraction of sim_q2 (teacher1 v.s. teacher1)

        AZ: the propagated fusion embedding AZ
        Z: the fusion embedding Z
    Returns:
        loss_DEG_DIFF
    """
    # sample correlation matrix
    ae1_deg = cross_correlation(sim_q1[0], sim_k1[0])
    gae1_deg = cross_correlation(sim_q1[2], sim_k1[2])

    ae1_diff = cross_correlation(sim_q1[1], sim_k1[1])
    gae1_diff = cross_correlation(sim_q1[3], sim_k1[3])

    ae2_deg = cross_correlation(sim_q2[0], sim_k2[0])
    gae2_deg = cross_correlation(sim_q2[2], sim_k2[2])

    ae2_diff = cross_correlation(sim_q2[1], sim_k2[1])
    gae2_diff = cross_correlation(sim_q2[3], sim_k2[3])

    # loss 
    L_ae_deg = correlation_reduction_loss(ae1_deg) + correlation_reduction_loss(ae2_deg)
    L_gae_deg = correlation_reduction_loss(gae1_deg) + correlation_reduction_loss(gae2_deg)
    L_ae_diff = correlation_reduction_loss(ae1_diff) + correlation_reduction_loss(ae2_diff)
    L_gae_diff = correlation_reduction_loss(gae1_diff) + correlation_reduction_loss(gae2_diff)


    loss_DEG_DIFF = 0.1 * L_ae_deg + 1 * L_gae_deg + 0.1 * L_ae_diff + 0.1 * L_gae_diff
    # loss_DEG_DIFF = 0.1 * L_ae_deg + 1 * L_gae_deg # global
    # loss_DEG_DIFF = 0.1 * L_ae_diff + 0.1 * L_gae_diff #local
    # loss_DEG_DIFF = 0 # bothno

    return loss_DEG_DIFF


def clustering(Z, y):
    """
    clustering based on embedding
    Args:
        Z: the input embedding
        y: the ground truth

    Returns: acc, nmi, ari, f1, clustering centers
    """
    model = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    cluster_id = model.fit_predict(Z.data.cpu().numpy())
    acc, nmi, ari, f1 = eva(y, cluster_id, show_details=opt.args.show_training_details)
    return acc, nmi, ari, f1, model.cluster_centers_


def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, show_details=False):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # if show_details:
    #     print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
    #           ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1