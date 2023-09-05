from train import *
from R2FGC import R2FGC
import opt

import scipy.sparse as sp
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data
import torch_geometric.transforms as T

torch.cuda.set_device(0)


if __name__ == '__main__':
    # setup
    setup()
    print(opt.args)

    # data pre-precessing: X, y, A, A_norm
    X, y, A = load_graph_data(opt.args.name, show_details=False)
    A_norm = normalize_adj(A, self_loop=True, symmetry=False)

    # to torch tensor
    X = numpy_to_torch(X)
    A_norm = numpy_to_torch(A_norm)

    tmp_coo = sp.coo_matrix(A) 
    edge_index, edge_attr = tg_utils.from_scipy_sparse_matrix(tmp_coo)
    data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
    print(f"Data: {data}")

    # Create Inverse Degree distribution
    degree = np.log(np.asarray(tg_utils.degree(data.edge_index[0], num_nodes = data.x.shape[0])) + 1)
    inv_degree = np.power(opt.args.alpha, degree).astype('float64')
    inv_degree[inv_degree > 1] = 1
    inv_degree /= inv_degree.sum()
    not_zero = np.where(np.asarray(tg_utils.degree(data.edge_index[0], num_nodes = data.x.shape[0])) != 0)[0]
    data_diffusion = T.GDC(sparsification_kwargs={'k': opt.args.topk, 'dim': 1, 'method': 'topk'})(data)
    diffusion = data_diffusion.edge_index[1].reshape(-1, opt.args.topk)

    # Relational Redundancy-Free Graph Clustering
    model = R2FGC(diffusion=diffusion, degree=inv_degree, not_zero=not_zero, \
        device=opt.args.device, n_node=data.x.size()[0]).to(opt.args.device)

    # deep graph clustering
    acc, nmi, ari, f1, best_epoch, best_Z, prelabel, run_time = train(model, data.to(opt.args.device), X, y, A_norm, A, opt.args.device)

    print("ACC: {:.4f},".format(acc), "NMI: {:.4f},".format(nmi), "ARI: {:.4f},".format(ari), "F1: {:.4f}".format(f1), \
        "BEST_EPOCH: {:.4f}".format(best_epoch))
    
    result_file = open("./results/result_table_seed.txt", mode="a", encoding="utf-8")
    result_file.write("%s"%opt.args.name + "\n")
    result_file.write("Seed:"+str(opt.args.seed)+"\n")
    result_file.write("ACC:"+str(acc)+"\n"+"NMI:"+str(nmi)+"\n"+"ARI:"+str(ari)+"\n"+"F1:"+str(f1)+"\n"+"BEST_EPOCH:"+str(best_epoch)+"\n")
    result_file.write("\n")
    result_file.close()
