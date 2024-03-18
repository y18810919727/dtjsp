import numpy as np
import torch.nn as nn
from L2D.mb_agg import *
# from models.mlp import MLPActor
# from models.mlp import MLPCritic
from .agent_utils import select_action
import torch.nn.functional as F
from decision_transformer.models.graphcnn_congForSJSSP import GraphCNN
from decision_transformer.models.decision_transformer import DecisionTransformer
from .mlp import MLPActor
import torch


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    # batch_size is the shape of batch
    # for graph pool sparse matrix
    if graph_pool_type == 'average':
        elem = torch.full(size=(batch_size[0]*n_nodes, 1),
                          fill_value=1 / n_nodes,
                          dtype=torch.float32,
                          device=device).view(-1)
    else:
        elem = torch.full(size=(batch_size[0] * n_nodes, 1),
                          fill_value=1,
                          dtype=torch.float32,
                          device=device).view(-1)
    idx_0 = torch.arange(start=0, end=batch_size[0],
                         device=device,
                         dtype=torch.long)
    # print(idx_0)
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size[0]*n_nodes, 1)).squeeze()

    idx_1 = torch.arange(start=0, end=n_nodes*batch_size[0],
                         device=device,
                         dtype=torch.long)
    idx = torch.stack((idx_0, idx_1))
    # graph_pool = torch.sparse.FloatTensor(idx, elem,
    #                                       torch.Size([batch_size[0],
    #                                                   n_nodes*batch_size[0]])
    #                                       ).to(device)
    graph_pool = torch.sparse_coo_tensor(
        idx, elem,
        torch.Size([batch_size[0], n_nodes*batch_size[0]]), device=device
    )

    return graph_pool


class dtg(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 # feature extraction net unique attributes:
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 num_mlp_layers_feature_extract,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # actor net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 device,
                 state_dim,
                 act_dim,
                 hidden_size,
                 g_pool_step='average',
                 max_length=None,
                 max_ep_len=4096,
                 n_positions=1024,
                 action_tanh=True,
                 **kwargs
                 ):
        super(dtg, self).__init__()
        # job size for problems, no business with network


        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device
        self.g_pool_step = g_pool_step
        self.max_length = max_length
        self.state_dim=state_dim
        self.act_dim=act_dim



        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        self.DecisionTransformer = DecisionTransformer(state_dim=state_dim,
                                                       act_dim=act_dim,
                                                       max_length=max_length,  # 最大序列长度
                                                       max_ep_len=max_ep_len,  # 最大episode长度，环境的最大步数
                                                       hidden_size=hidden_size,  # 隐藏层大小
                                                       n_positions=n_positions,
                                                       action_tanh=action_tanh)

        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*2, hidden_dim_actor, 1).to(device)



    def to_tensor(self, x):
        return torch.from_numpy(x).to(self.device)

    def gnn_encode_single_batch(self, x, adj, actions, candidate, last=False):
        """
        last:
            True: Only preserve the features of the available operations at the last step.
            False: Preserve the features of all operations, solely for training purposes.
        """
        # feat = torch.from_numpy(np.squeeze(np.array(x[i]))).to(self.device).reshape(-1,x[i].size(-1))
        episode_length, operation_size, _ = x.shape

        # batch_size = torch.Size([1, operation_size, operation_size])
        batch_size = adj.shape

        graph_pool = g_pool_cal(
            graph_pool_type=self.g_pool_step,
            batch_size=batch_size,
            n_nodes=operation_size,
            device=self.device
        )

        feat = x.reshape(-1, x.shape[-1])
        adj_matrix = aggr_obs(adj.to_sparse(), self.n_j*self.n_m)
        # candidate_tensor = to_tensor(candidate).squeeze(0)

        h_pooled, h_nodes = self.feature_extract(x=feat,
                                                 graph_pool=graph_pool,
                                                 padded_nei=None,
                                                 adj=adj_matrix)

        # h_pooled = h_pooled.reshape(episode_length, -1, h_pooled.size(-1))
        h_nodes = h_nodes.reshape(episode_length, operation_size, -1)

        # In the inferences stage, the action is equal to the num of tasks.
        # Add a dimention to h_nodes in dim=1 and fill it with 0,
        h_nodes = torch.cat([h_nodes, torch.zeros(episode_length, 1, h_nodes.size(-1), device=self.device)], dim=1)

        action_seq_feature = torch.gather(h_nodes, 1, actions.reshape([-1, 1, 1]).expand(-1, -1, h_nodes.size(-1))).squeeze(dim=1)

        dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
        if last:
            # In inference stage, only the features of the operations at the last step are preserved.
            dummy, h_nodes = dummy[-1], h_nodes[-1]

        # (episode length, num of available operations, feature size)
        candidate_features = torch.gather(h_nodes, -2, dummy)

        return h_pooled, action_seq_feature, candidate_features

    def prefix_padding(self, x: torch.Tensor, length, value=0):

        # x : (length, ...)
        if len(x.shape) == 1:
            return F.pad(x, (length - x.size(0), 0), mode='constant', value=value)
        else:
            return F.pad(x, (0, 0, length - x.size(0), 0), mode='constant', value=value)
        # return F.pad(x, (0, 0, 0, length - x.size(0)), value)

    def forward(self,
                x,
                adj,
                candidate,
                mask,
                actions,
                rewards,
                returns_to_go,
                done,
                si,
                ):

        batch_size = len(x)
        trajs = []
        aligned_trajs = []
        not_aligned_trajs = []

        episode_lengths = [x[i].shape[0] for i in range(batch_size)]
        for i in range(batch_size):
            episode_length = episode_lengths[i]
            h_pooled, action_seq_feature, candidate_features =  self.gnn_encode_single_batch(
                self.to_tensor(x[i]), self.to_tensor(adj[i]), self.to_tensor(actions[i]), self.to_tensor(candidate[i]), last=False
            )

            # Tensors that should be padded for alignment
            h_pooled = self.prefix_padding(h_pooled, self.max_length)
            action_seq_feature = self.prefix_padding(action_seq_feature, self.max_length)
            pad_rewards = self.prefix_padding(self.to_tensor(rewards[i]), self.max_length)
            pad_returns_to_go = self.prefix_padding(self.to_tensor(returns_to_go[i]), self.max_length)

            timesteps = self.prefix_padding(
                self.to_tensor(np.array(si[i])).unsqueeze(dim=-1) + torch.arange(episode_length).to(self.device), self.max_length
            )
            Attention_mask = torch.ones_like(timesteps)
            Attention_mask[:self.max_length-episode_length] = 0

            aligned_trajs.append((h_pooled, action_seq_feature, pad_rewards, pad_returns_to_go, timesteps, Attention_mask))


            # region packing tensors that will not be padded for alignment
            # ================================================
            # The following tensors are not utilized now

            # candidate_tensor = self.to_tensor(candidate[i])
            # mask_tensor = self.to_tensor(mask[i])
            # done_tensor = self.to_tensor(done[i])
            # ================================================

            # not_aligned_trajs.append((candidate_features, candidate_tensor, mask_tensor, done_tensor))
            not_aligned_trajs.append((candidate_features, ))
            # endregion

        # h_pooled, action_seq_feature, actions_fea_laststep, rewards, returns_to_go, candidate, mask, timesteps, done, Attention_mask = [torch.stack(x) for x in zip(*trajs)]
        h_pooled, action_seq_feature, rewards, returns_to_go, timesteps, Attention_mask = [torch.stack(x) for x in zip(*aligned_trajs)]

        # unpacking not_aligned_trajs
        # candidate_features, candidate, mask, done = [x for x in zip(*not_aligned_trajs)]
        candidate_features, = [x for x in zip(*not_aligned_trajs)]

        state_preds, action_preds, reward_preds = self.DecisionTransformer.forward(
            h_pooled, action_seq_feature, rewards, returns_to_go, timesteps, attention_mask=Attention_mask
        )

        pi_list = []
        for i in range(batch_size):
            episode_length = episode_lengths[i]
            # action_preds[i, -episode_length:]
            cf = candidate_features[i]
            action_pred_fea_repeated = action_preds[i, -episode_length:].unsqueeze(1).expand_as(cf)
            concateFea = torch.cat((cf, action_pred_fea_repeated), dim=-1)
            candidate_scores = self.actor(concateFea)
            pi = F.softmax(candidate_scores, dim=1).squeeze(dim=-1)
            pi_list.append(pi)

        return pi_list

        # concateFea = torch.cat((candidate_features, h_pooled_repeated), dim=-1)
        # # candidate_scores = self.actor(concateFea)
        # act_dim = len(candidate[0])
        #
        # # perform mask
        # # mask_reshape = mask.reshape(candidate_scores.size())
        # # candidate_scores[mask_reshape] = float('-inf')
        #
        # # pi = F.softmax(candidate_scores, dim=1)
        # # v = self.critic(h_pooled)
        # return state_preds, action_preds, action_fea, reward_preds,A_mask

    def get_action(self, x, adj, candidate, mask, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        x, adj, actions, candidate = x[-self.max_length:], adj[-self.max_length:], actions[-self.max_length:], candidate[-self.max_length:]

        states, actions_feat, candidate_features = self.gnn_encode_single_batch(
            x, adj, actions, candidate, last=True
        )

        states = states.reshape(1, -1, self.state_dim)
        actions_feat = actions_feat.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        returns_to_go = returns_to_go[:,-self.max_length:]

        timesteps = timesteps[:,-self.max_length:]

        # pad all tokens to sequence length
        attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
        states = torch.cat(
            [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
            dim=1).to(dtype=torch.float32)
        actions_feat = torch.cat(
            [torch.zeros((actions_feat.shape[0], self.max_length - actions_feat.shape[1], self.act_dim),
                         device=actions_feat.device), actions_feat],
            dim=1).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
            dim=1).to(dtype=torch.float32)
        timesteps = torch.cat(
            [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
            dim=1
        ).to(dtype=torch.long)
        # else:
        #     attention_mask = None

        # _, action_preds, return_preds = self.forward(
        #     states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        state_preds, action_preds, reward_preds = self.DecisionTransformer.forward(
            states, actions_feat, rewards, returns_to_go, timesteps, attention_mask=attention_mask
        )

        last_action_feat_pred = action_preds[0,-1]

        action_pred_fea_repeated = last_action_feat_pred.unsqueeze(0).expand_as(candidate_features)
        concateFea = torch.cat((candidate_features, action_pred_fea_repeated), dim=-1)
        candidate_scores = self.actor(concateFea)
        pi = F.softmax(candidate_scores, dim=0)

        chosen_operation, _ = select_action(pi, candidate[-1])
        return chosen_operation

