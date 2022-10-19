import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algo.redq_td3.model import Actor

class REDQTD3(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_width,
            action_space,
            lamba ,
            gamma=0.99,
            tau=0.005,
            actor_lr=3e-4,
            critic_lr=3e-4,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            num_nets=10,
            alpha=0.4,
            pretrain=True,
            use_q_min=True,  # if False: REQD; True: min over Qs
            device = 'cpu'
    ):
        super(REDQTD3, self).__init__()
        self.lamba = lamba
        self.max_action = float(action_space.high[0])
        self.actor = Actor(state_dim, action_dim, hidden_width,action_space).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # self.critic = Critic(state_dim, action_dim,hidden_width=hidden_width, num_nets=num_nets).to(device)
        # self.critic_target = copy.deepcopy(self.critic)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.num_nets = num_nets
        self.pretrain = pretrain
        self.use_q_min = use_q_min

        self.total_it = 0

    def choose_action(self, state, evaluate=False):
        # Return the action to interact with env.
        if len(state.shape) == 1:  # if no batch dim
            state = state.reshape(1, -1)

        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = state.to(self.device)

        with torch.no_grad():
            action = self.actor(state)
        return self.actor(state).detach().cpu().numpy()[0]

    def learn(self, batch_list,critic,critic_target,critic_optimizer,loss_mode = "basic",expert_batch_list = None):
        #for i in range(6): #多更新j
            self.total_it += 1
            # state, action, next_state, reward, not_done = data

            critic_loss_dict = self.critic_learn(batch_list,critic,critic_target,critic_optimizer)

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:
                self.actor_learn(batch_list,critic,loss_mode=loss_mode,expert_batch_list=expert_batch_list)
                # Update the frozen target models
                self.soft_update(critic_target, critic)
                self.soft_update(self.actor_target, self.actor)
            return {"critic_loss": critic_loss_dict['critic_loss'],
                    "critic": critic_loss_dict['critic'],
                    "td_error": critic_loss_dict['td_error']}

    def critic_learn(self, batch_list, critic, critic_target, critic_optimizer):
        state_batch = torch.Tensor(np.array(batch_list["state_list"])).to(self.device)
        next_state_batch = torch.Tensor(np.array(batch_list["next_state_list"])).to(self.device)
        action_batch = torch.Tensor(np.array(batch_list["action_list"])).to(self.device)
        reward_batch = torch.Tensor(np.array(batch_list["reward_list"])).to(self.device).unsqueeze(1)
        done_batch = torch.Tensor(np.array(batch_list["done_list"])).to(self.device).unsqueeze(1)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action_batch) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                    self.actor_target(next_state_batch) + noise
            ).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            if self.use_q_min and not self.pretrain:
                target_Qs = critic_target(next_state_batch, next_action)
                target_Q, _ = torch.min(target_Qs, dim=0)
            else:  # REDQ
                random_idx = np.random.permutation(self.num_nets)  # 随机选择critic网络
                target_Qs = critic_target(next_state_batch, next_action)[random_idx]
                target_Q1, target_Q2 = target_Qs[:2]

                target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q
        # Get current Q estimates
        current_Qs = critic(state_batch, action_batch)
        # Compute critic loss
        current_Qs = current_Qs.unsqueeze(0)
        td_error = (current_Qs - target_Q).mean().item()
        # print(td_error)
        critic_loss = F.mse_loss(current_Qs, target_Q)  # ensemble的critic同时进行更新
        # Optimize the critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        return {"critic_loss": critic_loss.item(),
                "critic": current_Qs[0].mean().item(),
                "td_error":td_error}
    def actor_learn(self,batch_list,critic,loss_mode,expert_batch_list = None):
        # Compute actor loss
        if loss_mode == 'basic' or loss_mode == 'bc_loss':
            state_batch = torch.Tensor(np.array(batch_list["state_list"])).to(self.device)
            action_batch = torch.Tensor(np.array(batch_list["action_list"])).to(self.device)
            pi = self.actor(state_batch)
            Q = critic(state_batch, pi).mean(0)
            # actor loss
            if loss_mode == 'basic':
                actor_loss = -Q.mean().mean()
            elif loss_mode == 'bc_loss':
                actor_loss = -Q.mean() / Q.abs().mean().detach() + self.lamba * F.mse_loss(pi, action_batch)
            # Compute actor losse
            #
            # actor_loss = -Q.mean().mean()
            # Optimize the actor
        elif loss_mode == 'expert_bc_loss':
            state_batch = torch.Tensor(np.array(batch_list["state_list"])).to(self.device)
            action_batch = torch.Tensor(np.array(batch_list["action_list"])).to(self.device)
            expert_state_batch = torch.Tensor(np.array(expert_batch_list["state_list"])).to(self.device)
            expert_action_batch = torch.Tensor(np.array(expert_batch_list["action_list"])).to(self.device)
            pi = self.actor(state_batch)
            expert_pi = self.actor(expert_state_batch)
            a = torch.cat([state_batch,expert_state_batch],dim=0)
            b = torch.cat([pi,expert_pi],dim=0)
            Q = critic(a, b).mean(0)
            actor_loss = -Q.mean() / Q.abs().mean().detach() + self.lamba * F.mse_loss(expert_pi, expert_action_batch)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    def soft_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    def hard_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)