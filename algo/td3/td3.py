import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from algo.td3.model import  QNetwork
from torch import nn
from algo.sac.replay_buffer import ReplayBuffer
from algo.sac.prioritized_buffer import PrioritizedReplay
from algo.td3.model import Actor
import copy
import numpy as np
"""
ref：https://github.com/pranz24/pytorch-soft-actor-critic
"""
class TD3(nn.Module):
    def __init__(self, state_dim,
            action_dim,
            action_space,
            hidden_width,
            buffer_size,
            priority_buffer = False,
            actor_lr = 1e-3,
            critic_lr = 1e-3,
            tau = 0.005,
            gamma = 0.99,

            policy_noise = 0.2,
            noise_clip = 0.5,

            target_update_interval = 2,
            device= 'cpu',
            ):
        super().__init__()
        self.priority_buffer = priority_buffer
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.target_update_interval = target_update_interval
        self.max_action = float(action_space.high[0])

        self.device = device

        self.critic = QNetwork(state_dim, action_dim, hidden_width).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr= critic_lr)

        self.critic_target = QNetwork(state_dim, action_dim,hidden_width).to(self.device)
        self.hard_update(self.critic_target, self.critic)

        self.actor = Actor(state_dim, action_dim, hidden_width, action_space).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

        if self.priority_buffer:
            self.replay_buffer = PrioritizedReplay(capacity = buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        self.updates = 0
        self.training_step = 0

    def choose_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).detach().cpu().numpy()[0]
    def learn(self, batch_list):
        qf1_loss, qf2_loss,td_error1,td_error2 = self.critic_learn(batch_list)
        if self.training_step % self.target_update_interval == 0:
            self.last_actor_loss = self.actor_learn(batch_list)
            self.soft_update(self.critic_target, self.critic)
            self.soft_update(self.actor_target, self.actor)
        self.training_step += 1
        loss_dict = {
            "td_error1":td_error1,
            "td_error2":td_error2,
            "qf1_loss": qf1_loss,
            "qf2_loss": qf2_loss,
            "actor_loss": self.last_actor_loss["actor_loss"],
        }
        return loss_dict

    def critic_learn(self, batch_list):
        state_batch = torch.Tensor(np.array(batch_list["state_list"])).to(self.device)
        next_state_batch = torch.Tensor(np.array(batch_list["next_state_list"])).to(self.device)
        action_batch = torch.Tensor(np.array(batch_list["action_list"])).to(self.device)
        reward_batch = torch.Tensor(np.array(batch_list["reward_list"])).to(self.device).unsqueeze(1)
        done_batch = torch.Tensor(np.array(batch_list["done_list"])).to(self.device).unsqueeze(1)

        with torch.no_grad():
            noise = (
                    torch.randn_like(action_batch) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state_batch) + noise
            ).clamp(-self.max_action, self.max_action)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = reward_batch + (1-done_batch) * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        if self.priority_buffer:
            index = batch_list["indices_list"]
            weights =  torch.Tensor(np.array(batch_list["weights"])).to(self.device)
            td_error1 = next_q_value.detach() - qf1  # ,reduction="none"
            td_error2 = next_q_value.detach() - qf2
            qf1_loss = 0.5 * (td_error1.pow(2) * weights).mean()
            qf2_loss = 0.5 * (td_error2.pow(2) * weights).mean()
            prios = abs(((td_error1 + td_error2) / 2.0 + 1e-5).squeeze())
            self.replay_buffer.update_priorities(index, prios.data.cpu().numpy())
        else:
            qf1_loss = F.mse_loss(qf1,
                                  next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2,
                                  next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        td_error1 = (next_q_value.detach() - qf1).mean()  # ,reduction="none"
        td_error2 = (next_q_value.detach() - qf2).mean()
        #print(td_error1.item())
        #print(td_error2.item())
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        return qf1_loss.item(),qf2_loss.item(),td_error1.item(),td_error2.item()

    # policy网络的更新
    def actor_learn(self, batch_list):
        state_batch = torch.Tensor(np.array(batch_list["state_list"])).to(self.device)
        # Compute actor losse
        actor_loss = -self.critic.Q1(state_batch, self.actor(state_batch)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return {"actor_loss":actor_loss.item()}
    def soft_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    def hard_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

