import numpy as np
import torch


def evaluate_episode_rtg(
        env,
        env_traj,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        ep_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    # state = env.reset(env_traj['env_paras'])
    adj, fea, candidate, mask = env.reset(env_traj['env_paras'])


    # append a np array (a,b) in the tail of array with shape (c, a, b)

    adj_tensor = torch.from_numpy(adj).to(device).unsqueeze(dim=0)
    fea_tensor = torch.from_numpy(fea).to(device).unsqueeze(dim=0)
    candidate_tensor = torch.from_numpy(candidate).to(device).unsqueeze(dim=0)
    mask_tensor = torch.from_numpy(mask).to(device).unsqueeze(dim=0)

    state_traj = [fea_tensor, adj_tensor, candidate_tensor, mask_tensor]

    actions = -torch.ones((0,), device=device, dtype=torch.int64)
    rewards = torch.zeros((0,), device=device, dtype=torch.float32)
    target_return = torch.zeros((0,), device=device, dtype=torch.float32)

    lb_makespan, episode_return, episode_length = env.max_endTime, 0, 0

    # target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    for t in range(max_ep_len):

        actions = torch.cat([actions, fea_tensor.shape[1] * torch.ones((1,), device=device, dtype=torch.int64)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device, dtype=torch.float32)], dim=0)
        target_return = torch.cat([target_return, torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1,)], dim=0)

        action = model.get_action(
            *state_traj,
            actions,
            rewards,
            target_return,
            timesteps
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        # state, reward, done, _ = env.step(action)
        adj, fea, reward, done, candidate, mask = env.step(action)

        # for traj, cur in zip([adj_tensor, fea_tensor, candidate_tensor, mask_tensor], [adj, fea, candidate, mask]):
        #     traj = torch.cat([traj, torch.from_numpy(cur).to(device).unsqueeze(dim=0)], dim=0)

        state_traj = [
            torch.cat([traj, torch.from_numpy(cur).to(device).unsqueeze(dim=0)], dim=0)
            for traj, cur in zip(state_traj, [fea, adj, candidate, mask])
        ]

        # substitute the last zero reward with the actual reward
        rewards[-1] = torch.tensor(reward, dtype=rewards.dtype).to(device)

        # Hindsight RTG update
        target_return = target_return+reward/scale

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1
        )

        episode_return += reward
        lb_makespan -= reward
        episode_length += 1

        if done:
            break

    return episode_return, lb_makespan, -env_traj['ep_reward'], episode_length
