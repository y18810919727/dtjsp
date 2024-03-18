import gym
import numpy as np
import torch
import wandb
import os

import pickle
import random
import sys

from config import args
from L2D.JSSP_Env import SJSSP

from decision_transformer.models.dtg import dtg

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def train(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']  # dataset = 6_6_1_99
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'jsp':
        env = SJSSP(n_j=variant['n_j'], n_m=variant['n_m'])
    else:
        raise ValueError(f'Unknown environment {env_name}')

    max_ep_len = variant.get('max_ep_len', 300)
    # max episode length
    env_target = [0] # 待定
    scale = variant.get('reward_scale', 300)


    trajectories = []
    for root, dirs, files in os.walk("L2D/pkl"):
        for file in files:
            if not os.path.splitext(file)[1] == '.pkl' or not file.startswith(f'{dataset}'):
                continue

            with open(os.path.join(root, file), 'rb') as f:
                pkl = pickle.load(f)
                trajectories += pkl

    if len(trajectories) == 0:
        raise ValueError(f'No trajectories found for {env_name} {dataset}')

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    traj_lens = np.array([len(traj['actions']) for traj in trajectories])
    returns = np.array([float(traj['ep_reward']) for traj in trajectories])
    num_timesteps = sum(traj_lens)   # step number

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    # print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    # print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    # context length of Decision Transformer
    K = variant['K']

    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=225, max_train_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        # fea_tensor_envs,adj_tensor_envs, candidate_tensor_envs, mask_tensor_envs, a, r, d, sim ,rtg=[], [], [], [], [], [], [],[],[]
        batch_trajs = []

        def cp(x):
            if isinstance(x, np.ndarray):
                return np.copy(x)
            else:
                return x

        max_len = max([len(traj['actions']) for traj in trajectories])
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]

            env = SJSSP(n_j=traj['N_JOBS_P'], n_m=traj['N_MACHINES_P'])
            adj, fea, candidate, mask = env.reset(traj['env_paras'])
            traj_len = len(traj['actions'])

            # complete_traj = [(adj, fea, candidate, mask, 0, False)]
            # state_traj = [(adj, fea, candidate, mask)]
            adj_traj = [cp(adj)]
            fea_traj = [cp(fea)]
            candidate_traj = [cp(candidate)]
            mask_traj = [cp(mask)]

            reward_traj = []
            action_traj = []
            done_traj = []


            for action in traj['actions']:
                assert action in candidate
                adj, fea, reward, done, candidate, mask = env.step(action)
                for a, b in zip(
                        [adj_traj, fea_traj, candidate_traj, mask_traj, reward_traj, action_traj, done_traj],
                        [adj, fea, candidate, mask, reward, action, done]
                ):
                    a.append(cp(b))

                if done:
                    break
            # split complete_traj into multiple trajectories
            # _, _, _, _, rewards, _ = zip(*complete_traj)
            rtgs_traj = discount_cumsum(np.array(reward_traj, dtype=np.float32), gamma=1.0)/ scale

            si = random.randint(0, max(0, traj_len - max_train_len//2))
            begin, end = si, si + max_train_len
            # state_traj, reward_traj, rtgs_traj, action_traj, done_traj = [np.array(x)[begin:end] for x in [state_traj, reward_traj, rtgs_traj, action_traj, done_traj]]
            adj_traj, fea_traj, candidate_traj, mask_traj, reward_traj, rtgs_traj, action_traj, done_traj = [
                np.array(x)[begin:end] for x in [adj_traj[:-1], fea_traj[:-1], candidate_traj[:-1], mask_traj[:-1], reward_traj, rtgs_traj, action_traj, done_traj]
            ]
            # expand last dim of array
            # state_traj = np.expand_dims(state_traj, -1)

            def expand(x):
                return np.expand_dims(x, -1)
            # segments_traj.append(rtgs[begin:end])
            batch_trajs.append((adj_traj, fea_traj, candidate_traj, mask_traj, expand(reward_traj), expand(rtgs_traj), action_traj, done_traj, si))


        adj_trajs, fea_trajs, candidate_trajs, mask_trajs, reward_trajs, rtgs_trajs, action_trajs, done_traj, sis = [list(x) for x in zip(*batch_trajs)]
        return adj_trajs, fea_trajs, candidate_trajs, mask_trajs, reward_trajs, rtgs_trajs, action_trajs, done_traj, sis

    def eval_episodes(target_rew, trajectories, env):
        def fn(model):
            makespans, base_makespans, returns, lengths = [], [], [], []
            for _ in range(num_eval_episodes):
                # Choose a trajectory from trajectories randomlly
                env_traj = random.choice(trajectories)
                with torch.no_grad():
                    ret, makespan, behavior_makespan, length = evaluate_episode_rtg(
                        env,
                        env_traj,
                        model,
                        max_ep_len=env_traj['N_JOBS_P'] * env_traj['N_MACHINES_P'],
                        scale=scale,
                        ep_return=target_rew / scale,
                        mode=mode,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
                makespans.append(makespan)
                base_makespans.append(behavior_makespan)
            return {
                f'dtjsp_makespan_mean': np.mean(makespans),
                f'dtjsp_makespan_std': np.std(makespans),
                f'base_makespan_mean': np.mean(behavior_makespan),
                f'base_makespan_std': np.std(behavior_makespan),
                f'return_mean': np.mean(returns),
                f'return_std': np.std(returns),
            }
        return fn

    model = dtg(
        n_j = variant['n_j'],
        n_m = variant['n_m'],
        g_pool_step = variant['graph_pool_type'],
        # feature extraction net unique attributes:
        num_layers = variant['num_layers'],
        learn_eps = False,
        neighbor_pooling_type = variant['neighbor_pooling_type'],
        input_dim = 2,
        hidden_dim = variant['hidden_dim'],
        # feature extraction net MLP attributes:
        num_mlp_layers_feature_extract = variant['num_mlp_layers_feature_extract'],
        # actor net MLP attributes:
        num_mlp_layers_actor = variant['num_mlp_layers_actor'],
        hidden_dim_actor = variant['hidden_dim_actor'],
        # actor net MLP attributes:
        num_mlp_layers_critic = variant['num_mlp_layers_critic'],
        hidden_dim_critic = variant['hidden_dim_critic'],
        # actor/critic/feature_extraction shared attribute
        device = device,
        state_dim=variant['hidden_dim'],
        act_dim=variant['hidden_dim'],
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
        variant=variant,
    )

    model = model.to(device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=[eval_episodes(tar, trajectories, env) for tar in env_target],
    )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )


    for iter in range(variant['max_iters']):
        print('Iter: ', iter)
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)

if __name__ == '__main__':

    train('DTG', variant=vars(args))






