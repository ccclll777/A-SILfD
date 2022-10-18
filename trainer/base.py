
from algo.td3.td3 import TD3
from algo.redq_td3.redq_td3 import REDQTD3

def get_td3_agent(args,td3_configs):
    td3_agent = TD3(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        action_space = args.action_space,
        hidden_width=td3_configs.hidden_width,
        buffer_size = td3_configs.buffer_size,
        priority_buffer=td3_configs.priority_buffer,

        actor_lr=td3_configs.actor_lr,
        critic_lr=td3_configs.critic_lr,
        tau = td3_configs.tau,
        gamma = td3_configs.gamma,
        policy_noise = td3_configs.policy_noise,
        noise_clip = td3_configs.noise_clip,
        target_update_interval = td3_configs.target_update_interval,
        device = args.device )
    return td3_agent

def get_redq_td3_agent(args,td3_configs):
    redq_td3_agent = REDQTD3(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        action_space = args.action_space,
        lamba=args.lamba,
        hidden_width=td3_configs.hidden_width,
        actor_lr=td3_configs.actor_lr,
        critic_lr=td3_configs.critic_lr,
        tau = td3_configs.tau,
        gamma = td3_configs.gamma,
        policy_noise = td3_configs.policy_noise,
        noise_clip = td3_configs.noise_clip,
        device = args.device )
    return redq_td3_agent


