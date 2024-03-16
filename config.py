#!/usr/bin/python
# -*- coding:utf8 -*-
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='jsp')
parser.add_argument('--dataset', type=str, default='DRL')  # medium, medium-replay, medium-expert, expert
parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
parser.add_argument('--K', type=int, default=200)
parser.add_argument('--reward_scale', type=int, default=100)
parser.add_argument('--pct_traj', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
parser.add_argument('--embed_dim', type=int, default=768)
parser.add_argument('--n_layer', type=int, default=3)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--activation_function', type=str, default='relu')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--num_eval_episodes', type=int, default=5)
parser.add_argument('--max_iters', type=int, default=30)
parser.add_argument('--num_steps_per_iter', type=int, default=100)
# parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
parser.add_argument('--device', type=str, default="cuda", help='devices')
# args for env
parser.add_argument('--n_j', type=int, default=15, help='Number of jobs of instance')
parser.add_argument('--n_m', type=int, default=15, help='Number of machines instance')
parser.add_argument('--rewardscale', type=float, default=0., help='Reward scale for positive rewards')
parser.add_argument('--init_quality_flag', type=bool, default=False,
                    help='Flag of whether init state quality is 0, True for 0')
parser.add_argument('--low', type=int, default=1, help='LB of duration')
parser.add_argument('--high', type=int, default=99, help='UB of duration')
parser.add_argument('--np_seed_train', type=int, default=200, help='Seed for numpy for training')
parser.add_argument('--np_seed_validation', type=int, default=200, help='Seed for numpy for validation')
parser.add_argument('--torch_seed', type=int, default=600, help='Seed for torch')
parser.add_argument('--et_normalize_coef', type=int, default=1000,
                    help='Normalizing constant for feature LBs (end time), normalization way: fea/constant')
parser.add_argument('--wkr_normalize_coef', type=int, default=100,
                    help='Normalizing constant for wkr, normalization way: fea/constant')
# args for network
parser.add_argument('--num_layers', type=int, default=3,
                    help='No. of layers of feature extraction GNN including input layer')
parser.add_argument('--neighbor_pooling_type', type=str, default='sum', help='neighbour pooling type')
parser.add_argument('--graph_pool_type', type=str, default='average', help='graph pooling type')
parser.add_argument('--max_ep_len', type=int, default=300, help='max length of episode in training')
parser.add_argument('--input_dim', type=int, default=2, help='number of dimension of raw node features')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dim of MLP in fea extract GNN')
parser.add_argument('--num_mlp_layers_feature_extract', type=int, default=2,
                    help='No. of layers of MLP in fea extract GNN')
parser.add_argument('--num_mlp_layers_actor', type=int, default=2, help='No. of layers in actor MLP')
parser.add_argument('--hidden_dim_actor', type=int, default=32, help='hidden dim of MLP in actor')
parser.add_argument('--num_mlp_layers_critic', type=int, default=2, help='No. of layers in critic MLP')
parser.add_argument('--hidden_dim_critic', type=int, default=32, help='hidden dim of MLP in critic')

args = parser.parse_args()







