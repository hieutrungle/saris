import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common import buffers
from saris.utils import pytorch_utils


def eval_trajectory(
    self, agent: nn.Module, envs: gym.vector.VectorEnv, eval_ep_len: int, seed: int
):
    obs, _ = envs.reset(seed=seed)
    num_envs = len(obs)

    buffer = buffers.DictReplayBuffer(
        buffer_size=eval_ep_len,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=torch.device("cpu"),
        n_envs=num_envs,
    )

    path_gains = np.full(eval_ep_len, num_envs, np.nan)

    # TODO: check the output/shape of obs
    for step in range(eval_ep_len):
        obs = pytorch_utils.add_batch_dim(obs)
        observations = pytorch_utils.from_numpy(observations, self.device)
        actions = self.agent.get_actions(observations)
        actions = pytorch_utils.to_numpy(actions)
        action = np.squeeze(actions, axis=0)
        try:
            next_ob, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"Error in step {step}: {e}")
            time.sleep(2)
            continue

        done = terminated or truncated
        done = np.asarray(done, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32)
        obs[step] = ob
        acts[step] = action
        rews[step] = reward
        next_obs[step] = next_ob
        dones[step] = done
        path_gains[step] = info["path_gain_dB"]

        if done:
            (ob, info) = env.reset()
            break
        else:
            ob = next_ob

    return {
        "obs": obs,
        "acts": acts,
        "rews": rews,
        "next_obs": next_obs,
        "dones": dones,
        "path_gains": path_gains,
        "return": np.nansum(rews),
    }


def eval_trajectories(
    self, env: gym.Env, num_evals: int = 3, eval_ep_len: int = 30
) -> Sequence[Dict[str, np.ndarray]]:
    trajectories = []
    env.unwrapped.eval = True
    for i in range(num_evals):
        trajectory = self.eval_trajectory(env, eval_ep_len)
        trajectories.append(trajectory)
    env.unwrapped.eval = False

    return trajectories
