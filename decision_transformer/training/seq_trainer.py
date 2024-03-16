import numpy as np
import torch

from decision_transformer.training.trainer import Trainer
import torch.nn.functional as F

from ..models.agent_utils import eval_actions


class SequenceTrainer(Trainer):

    def train_step(self):
        # fea_tensor_envs, g_pool_step, adj_tensor_envs, candidate_tensor_envs, mask_tensor_envs, a, r, d, rtg, si, max_len= self.get_batch(self.batch_size)

        adj_trajs, fea_trajs, candidate_trajs, mask_trajs, reward_trajs, rtgs_trajs, action_trajs, done_traj, sis = self.get_batch(self.batch_size)

        # state_preds, action_preds, action_fea, reward_preds,A_mask = self.model.forward(
        #     # fea_tensor_envs, g_pool_step, adj_tensor_envs, candidate_tensor_envs, mask_tensor_envs, a, r, d,rtg, si,max_len
        #     fea_trajs, adj_trajs, candidate_trajs, mask_trajs, action_trajs, reward_trajs, rtgs_trajs, done_traj, sis
        # )

        pi_list = self.model.forward(
            # fea_tensor_envs, g_pool_step, adj_tensor_envs, candidate_tensor_envs, mask_tensor_envs, a, r, d,rtg, si,max_len
            fea_trajs, adj_trajs, candidate_trajs, mask_trajs, action_trajs, reward_trajs, rtgs_trajs, done_traj, sis
        )

        loss = 0
        for i, (pi, action, candidate) in enumerate(zip(pi_list, action_trajs, candidate_trajs)):
            # action: (n) , candidate: np.array with shape (n, M), each subarray with size M is a array of indices. Find the indices of action in candidate
            try:
                action_indices = [np.where(candidate[i] == action[i])[0][0] for i in range(len(action))]
            except Exception as e :
                pos = 0
                for pos in range(len(action)):
                    if int(action[pos]) not in candidate[pos]:
                        print(pos)
                        break
                print(action[pos], candidate[pos])
                raise e
            # action_indices = [candidate[i].index(action[i]) for i in range(len(action))]
            loss += F.nll_loss(F.log_softmax(pi, dim=1), torch.tensor(action_indices).unsqueeze(dim=-1).to(pi.device))
            # logprobs, ent_loss = eval_actions(pi, action)
            # loss += -ent_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/entropy_loss'] = loss.detach().cpu().item()

        return loss.detach().cpu().item()
