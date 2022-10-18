from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import torch
class DemonstrateDataset(Dataset):
    def  __init__(self,file_path,device = 'cpu'):
        """
        https://github.com/takuseno/d4rl-pybullet
        https://blog.csdn.net/gsww404/article/details/123802410
        :param task: 任务 hopper  ant half walker2d
        :param level: random
                         medium-replay
        :param        medium-expert
        :param        medium
        """
        self.device = device
        init_dataset = pickle.load(open(file_path, 'rb'))
        dataset = dict()
        dataset["states"] = init_dataset["states"]
        dataset["next_states"] = init_dataset["next_states"]
        dataset["actions"] = init_dataset["actions"]
        dataset["rewards"] = init_dataset["rewards"]
        dataset["dones"] = init_dataset["dones"]
        #可用的专家数据
        self.expert_states =  dataset["states"]
        self.expert_next_states = dataset["next_states"]
        self.expert_actions = dataset["actions"]
        self.expert_rewards = dataset["rewards"]
        self.expert_dones = dataset["dones"]


        # Input data
        self.source_state = dataset["states"][:-1]
        self.source_action = dataset["actions"][:-1]
        # Output data
        self.target_delta = dataset["states"][1:] - self.source_state
        # Normalize data
        self.delta_mean = self.target_delta.mean(axis=0)
        self.delta_std = self.target_delta.std(axis=0)
        self.state_mean = self.source_state.mean(axis=0)
        self.state_std = self.source_state.std(axis=0)
        self.action_mean = self.source_action.mean(axis=0)
        self.action_std = self.source_action.std(axis=0)
        self.source_action = (self.source_action - self.action_mean)/self.action_std
        self.source_state = (self.source_state - self.state_mean)/self.state_std
        self.target_delta = (self.target_delta - self.delta_mean)/self.delta_std
        # Get indices of initial states #初始状态的索引
        self.done_indices = dataset["dones"][:-1]
        self.initial_indices = np.roll(self.done_indices, 1)
        self.initial_indices[0] = True
        # Calculate distribution parameters for initial states
        self.initial_obs = self.source_state[self.initial_indices]
        self.initial_obs_mean = self.initial_obs.mean(axis = 0)
        self.initial_obs_std = self.initial_obs.std(axis = 0)

        # Remove transitions from terminal to initial states #剔除终止状态到起始状态的transition
        self.source_action = np.delete(self.source_action, self.done_indices, axis = 0)
        self.source_state = np.delete(self.source_state, self.done_indices, axis = 0)
        self.target_delta = np.delete(self.target_delta, self.done_indices, axis = 0)
    def save_data(self):
        dump = []
        dump.append(self.state_mean)
        dump.append(self.state_std)
        dump.append(self.action_mean)
        dump.append(self.action_std)
        with open('meanstd','wb') as f:
            pickle.dump(dump, f)
    def __getitem__(self, idx):
        input = torch.FloatTensor(np.concatenate([self.source_state[idx], self.source_action[idx]])).to(self.device)
        target = torch.FloatTensor(self.target_delta[idx]).to(self.device)
        return input, target
    def __len__(self):
        return len(self.source_state)

class Sampling:
    def __init__(self):
        self.actions = []
        self.states = []
        self.done = []

    def push(self, state, action, done):
        self.states.append(state)
        self.actions.append(action)
        self.done.append(done)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.done[:]

    def size(self):
        return len(self.states)


class SamplingDataset(Dataset):
    def __init__(self, states, actions, dones, device='cpu'):
        """
        这个是从环境中采样的数据，用于训练dynmaic model
        :param states:
        :param actions:
        :param dones:
        :param device:
        """
        self.device = device
        # states = torch.squeeze(torch.stack(states, dim=0)).detach().cpu()
        # actions = torch.squeeze(torch.stack(actions, dim=0)).detach().cpu()
        dataset = dict()
        dataset["states"] = np.array(states)
        dataset["actions"] = np.array(actions)
        dataset["dones"] = np.array(dones)
        # Input data
        self.source_state = dataset["states"][:-1]
        self.source_action = dataset["actions"][:-1]
        # Output data
        self.target_delta = dataset["states"][1:] - self.source_state
        # Normalize data
        self.delta_mean = self.target_delta.mean(axis=0)
        self.delta_std = self.target_delta.std(axis=0)
        self.state_mean = self.source_state.mean(axis=0)
        self.state_std = self.source_state.std(axis=0)

        self.action_mean = self.source_action.mean(axis=0)
        self.action_std = self.source_action.std(axis=0)

        self.source_action = (self.source_action - self.action_mean) / self.action_std
        self.source_state = (self.source_state - self.state_mean) / self.state_std
        self.target_delta = (self.target_delta - self.delta_mean) / self.delta_std

        # Get indices of initial states #初始状态的索引
        self.done_indices = dataset["dones"][:-1]
        self.initial_indices = np.roll(self.done_indices, 1)
        self.initial_indices[0] = True

        # Calculate distribution parameters for initial states
        self.initial_obs = self.source_state[self.initial_indices]
        self.initial_obs_mean = self.initial_obs.mean(axis=0)
        self.initial_obs_std = self.initial_obs.std(axis=0)

        # Remove transitions from terminal to initial states #剔除终止状态到起始状态的transition
        self.source_action = np.delete(self.source_action, self.done_indices, axis=0)
        self.source_state = np.delete(self.source_state, self.done_indices, axis=0)
        self.target_delta = np.delete(self.target_delta, self.done_indices, axis=0)

    def save_data(self):
        dump = []
        dump.append(self.state_mean)
        dump.append(self.state_std)
        dump.append(self.action_mean)
        dump.append(self.action_std)
        with open('meanstd', 'wb') as f:
            pickle.dump(dump, f)

    def __getitem__(self, idx):
        input = torch.FloatTensor(np.concatenate([self.source_state[idx], self.source_action[idx]])).to(self.device)
        target = torch.FloatTensor(self.target_delta[idx]).to(self.device)
        return input, target

    def __len__(self):
        return len(self.source_state)
def weight_decay(step,weight_init =2 ,decay_rate = 0.000005 ):
    """
    权重衰减方法
    :param step:  当前时间步
    :param weight_init:  初始化权重  0.4时 1000个episode会衰减到很小
    :param decay_rate:  衰减率

    :return:
    """
    return weight_init * ((1 - decay_rate) ** step)

def evaluate(env, agent, eval_episode, episode_max_steps,total_steps = 0,writer = None,):
    avg_reward = 0.
    avg_length = 0
    for _ in range(eval_episode):
        obs, done = env.reset(), False
        step = 0
        while not done and step <= episode_max_steps:
            action = agent.choose_action(obs,evaluate=True)

            next_obs, reward, done, info = env.step(action)
            avg_reward += reward
            obs = next_obs
            step += 1
        avg_length += step
    avg_reward /= eval_episode
    avg_length /= eval_episode
    if writer != None:
        writer.add_scalar("evaluate/reward", avg_reward, total_steps)
        writer.add_scalar("evaluate/length", avg_length, total_steps)
    print("evaluate/step:", total_steps, "steps", "average_reward", avg_reward)
    print("evaluate/step:", total_steps, "steps", "average_length", avg_length)
    return avg_reward,avg_length

def reward_weight_decay(step,weight_init =2 ,decay_rate = 0.0000005):
    return weight_init * ((1 - decay_rate) ** step)


def lamba_weight_decay(step,weight_init =0.4 ,decay_rate = 0.00001):
    return weight_init * ((1 - decay_rate) ** step)

def lamba_weight_decay_hard(weight =0.8 ,decay_rate = 0.02):
    if  weight - decay_rate <= 0:
        return 0
    return  round(weight - decay_rate, 4)


def kl_divergence(p_logs, q_logs):
	"""
	Compute KL divergence between log policies
	:param p_logs: pytorch tensor of shape (N, |A|) with N = batchsize and |A| = nmber of actions
		including log-probs for each state of batch and each action
	:param q_logs: pytorch tensor of shape (N, |A|) with N = batchsize and |A| = nmber of actions
		including log-probs for each state of batch and each action
	:return: pytorch tensor of shape (N,) with KL divergence KL(p || q) for each batch
	"""
	kl = (p_logs.exp() * (p_logs - q_logs)).sum(-1)
	return kl