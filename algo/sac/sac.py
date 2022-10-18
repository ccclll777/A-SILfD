import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from algo.sac.model import GaussianPolicy, QNetwork, DeterministicPolicy
from torch import nn
from algo.sac.replay_buffer import ReplayBuffer
from algo.sac.prioritized_buffer import PrioritizedReplay
import numpy as np
"""
ref：https://github.com/pranz24/pytorch-soft-actor-critic
"""
class SAC(nn.Module):
    def __init__(self, state_dim,
            action_dim,
            action_space,
            hidden_width,
            buffer_size,
            priority_buffer = False,
            policy_type ="Gaussian",
            actor_lr = 1e-3,
            critic_lr = 1e-3,
            tau = 0.005,
            gamma = 0.99,
            alpha = 0.2,
            alpha_lr = 0.001,
            automatic_entropy_tuning:bool = True,
            target_update_interval = 2,
            device= 'cpu',
            ):
        super().__init__()
        self.lamba = 0.4
        self.priority_buffer = priority_buffer
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.policy_type = policy_type
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.critic_lr = critic_lr
        self.critic = QNetwork(state_dim, action_dim, hidden_width).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr= critic_lr)

        self.critic_target = QNetwork(state_dim, action_dim,hidden_width).to(self.device)
        self.hard_update(self.critic_target, self.critic)
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=alpha_lr)

            self.policy = GaussianPolicy(state_dim, action_dim, hidden_width, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(state_dim, action_dim,hidden_width, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)
        if self.priority_buffer:
            self.replay_buffer = PrioritizedReplay(capacity = buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        self.updates = 0 #policy的更新次数
        self.training_step = 0
    def choose_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    def learn(self, batch_list,loss_mode = ""):
        qf1_loss, qf2_loss = self.critic_learn(batch_list)
        if self.training_step % self.target_update_interval == 0:
            self.last_actor_loss = self.policy_learn(batch_list,loss_mode = loss_mode)
            self.soft_update(self.critic_target, self.critic)
        self.training_step += 1
        loss_dict = {
            "qf1_loss": qf1_loss,
            "qf2_loss": qf2_loss,
            "policy_loss": self.last_actor_loss["policy_loss"],
            "policy_entropy": self.last_actor_loss["policy_entropy"],
            "alpha_loss":self.last_actor_loss["alpha_loss"],
            "alpha_tlogs":self.last_actor_loss["alpha_tlogs"],
        }
        return loss_dict

    def critic_learn(self, batch_list):
        state_batch = torch.Tensor(np.array(batch_list["state_list"])).to(self.device)
        next_state_batch = torch.Tensor(np.array(batch_list["next_state_list"])).to(self.device)
        action_batch = torch.Tensor(np.array(batch_list["action_list"])).to(self.device)
        reward_batch = torch.Tensor(np.array(batch_list["reward_list"])).to(self.device).unsqueeze(1)
        done_batch = torch.Tensor(np.array(batch_list["done_list"])).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
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
                                  next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        return qf1_loss.item(),qf2_loss.item()

    # policy网络的更新
    def policy_learn(self, batch_list,loss_mode = ""):
        state_batch = torch.Tensor(np.array(batch_list["state_list"])).to(self.device)
        action_batch = torch.Tensor(np.array(batch_list["action_list"])).to(self.device)
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)


        if loss_mode == 'bc_loss':
            bc_loss = self.lamba * F.mse_loss(pi, action_batch)
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() + bc_loss
        elif loss_mode == 'adv':
            beta = 2
            qf1_old, qf2_old = self.critic(state_batch, action_batch)
            min_q_old = torch.min(qf1_old, qf2_old)

            adv_pi = min_q_old - min_qf_pi
            #weights = torch.exp(adv_pi)
            weights = F.softmax(adv_pi / beta, dim=0)
            policy_loss = (-log_pi * len(weights) * weights.detach()).mean()
        else:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()# Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs
        return {"policy_loss":policy_loss.item(),
                "policy_entropy":np.mean(log_pi.detach().cpu().numpy()),
                "alpha_loss":alpha_loss.item(),
                "alpha_tlogs":alpha_tlogs.item()}
    def soft_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    def hard_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

