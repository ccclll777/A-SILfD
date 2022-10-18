import gym
from trainer.base import get_redq_td3_agent,get_td3_agent
import numpy as np
import argparse
from utils.trainer_utils import evaluate
from algo.sac.replay_buffer import ReplayBuffer
from algo.redq_td3.model import Critic
import torch
import copy
from utils.trainer_utils import DemonstrateDataset
def init_replay_buffer(replay_buffer,demonstrates_data):
    expert_states = demonstrates_data.expert_states
    expert_actions = demonstrates_data.expert_actions
    expert_rewards = demonstrates_data.expert_rewards
    expert_next_states = demonstrates_data.expert_next_states
    expert_dones = demonstrates_data.expert_dones
    for i in range(1,len(expert_states)+1):
        replay_buffer.push(expert_states[i-1],expert_actions[i-1],expert_rewards[i-1],expert_next_states[i-1],expert_dones[i-1])
def redq_td3_trainer(args,configs,train_envs,eval_envs):
    args.lamba = 0.0
    envs = train_envs
    redq_td3 = get_redq_td3_agent(args,argparse.Namespace(**configs.td3))
    critic = Critic(args.state_dim, args.action_dim, hidden_width=configs.td3['hidden_width'], num_nets=10).to(args.device)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=configs.td3['critic_lr'])
    replay_buffer = ReplayBuffer(configs.td3['buffer_size'])
    trainning_args = argparse.Namespace(**configs.misc)

    if configs.ours['bc_pre_train']:
       redq_td3.actor.load_state_dict(torch.load(configs.ours['bc_model_path'],
                      map_location=args.device))
    if configs.redq['pretrain_demo']: #用LfD的方式进行pre train
        demonstrates_data = DemonstrateDataset(
            file_path=configs.env['demonstrate_path'],
            device=args.device)
        init_replay_buffer(replay_buffer,demonstrates_data)
        for i in range(configs.redq['pretrain_epoch']):
            batch_list = replay_buffer.sample(configs.td3['batch_size'])
            loss_dict = redq_td3.learn(batch_list, critic, critic_target, critic_optimizer)

    total_steps = 0
    #开始的评估
    evaluate(eval_envs, redq_td3, trainning_args.evaluate_episode,
                                              trainning_args.episode_max_steps, total_steps, args.writer)

    max_eval_avg_reward =  -2000


    evaluate_step = 0
    for episode in range(trainning_args.episodes):
        state = envs.reset()
        done = False
        episode_total_reward = 0
        steps = 0
        while not done and trainning_args.episode_max_steps >= steps:
            #bcc初始化之后也要随机探索
            if total_steps < configs.td3['start_steps'] :
                action = train_envs.action_space.sample()
            else:
                # Sample actions
                action = (
                        redq_td3.choose_action(state)
                        + np.random.normal(0, redq_td3.max_action * configs.td3['expl_noise'], size=args.action_dim)
                ).clip(-redq_td3.max_action, redq_td3.max_action)
            # Obser reward and next obs
            next_state, reward, done, infos = envs.step(action)
            episode_total_reward += reward
            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > configs.td3['batch_size']:
                batch_list = replay_buffer.sample(configs.td3['batch_size'])
                loss_dict = redq_td3.learn(batch_list,critic,critic_target,critic_optimizer)

            if args.render:
                envs.render()
            steps += 1
            total_steps += 1
            evaluate_step +=1
            state = next_state
        if total_steps >= configs.misc['num_steps']:
            break
        """
        每个episode之后打印log
        """

        if args.writer != None:
            args.writer.add_scalar("train/episode_reward", episode_total_reward, episode)
            args.writer.add_scalar("train/episode_length", steps, episode)
        # if episode % trainning_args.evaluate_freq == 0:
        #     print("train/episode:", episode, "reward：", episode_total_reward)
        #     print("train/episode:", episode, "length：", steps)
        """
        评估结果
        """
        # if episode % trainning_args.evaluate_freq == 0:
        if evaluate_step >= trainning_args.evaluate_freq_steps :
            print("train/episode:", episode, "reward：", episode_total_reward)
            print("train/episode:", episode, "length：", steps)
            evaluate_step = 0
            # 评估结果
            average_reward, average_length = evaluate(eval_envs, redq_td3, trainning_args.evaluate_episode,
                                                      trainning_args.episode_max_steps, total_steps, args.writer)

            if max_eval_avg_reward < average_reward and args.save and args.train:
                max_eval_avg_reward = average_reward
                torch.save(redq_td3.state_dict(),
                           args.model_path + f"/reqd_td3_steps{total_steps}_reward{average_reward:.0f}.pth")

