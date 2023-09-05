import argparse

parser = argparse.ArgumentParser(description='R2FGC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default="acm")
parser.add_argument('--n_clusters', type=int, default=3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--show_training_details', type=bool, default=False)
# Number of samples
parser.add_argument("--sample", type=int, default=256, help="The number of global sample. Default is 256")
parser.add_argument("--topk", type=int, default=8, help="The number of local sample. Default is 8")
# Hyperparameters for inverse degree sampling distribution
parser.add_argument("--alpha", type=float, default=0.8, help="Hyperparameters for the skewness of inverse degree sampling distribution")
# Hyperparameters for loss function
parser.add_argument('--kappa_value', type=int, default=10, help="controls the weight of clustering")
parser.add_argument('--epsilon_value', type=float, default=5e3)
# Other
parser.add_argument('--n_input', type=int, default=100)
parser.add_argument('--eta_value', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_z', type=int, default=20)

# AE structure parameter from DFCN
parser.add_argument('--ae_n_enc_1', type=int, default=128)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=512)
parser.add_argument('--ae_n_dec_1', type=int, default=512)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=128)

# IGAE structure parameter from DFCN
parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=256)
parser.add_argument('--gae_n_enc_3', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)

# clustering performance: acc, nmi, ari, f1
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--f1', type=float, default=0)

args = parser.parse_args()
