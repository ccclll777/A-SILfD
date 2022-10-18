import gym
from trainer.base import get_redq_td3_agent,get_td3_agent
import numpy as np
import argparse
from utils.trainer_utils import evaluate,lamba_weight_decay,lamba_weight_decay_hard
from algo.sac.replay_buffer import ReplayBuffer
import torch

from torch.utils.data import DataLoader
from utils.trainer_utils import DemonstrateDataset
from algo.redq_td3.model import Critic
import copy
def init_teacher_replay_buffer(teacher_replay_buffer,demonstrates_data):
    expert_states = demonstrates_data.expert_states
    expert_actions = demonstrates_data.expert_actions
    expert_rewards = demonstrates_data.expert_rewards
    expert_next_states = demonstrates_data.expert_next_states
    expert_dones = demonstrates_data.expert_dones
    expert_scores = []
    expert_score = 0
    for i in range(1,len(expert_states)+1):
        expert_score += expert_rewards[i-1]
        teacher_replay_buffer.push(expert_states[i-1],expert_actions[i-1],expert_rewards[i-1],expert_next_states[i-1],expert_dones[i-1])
        # if i % 1000 == 0:
        if expert_dones[i-1]:
            expert_scores.append(expert_score)
            expert_score = 0
    if expert_score != 0:
        expert_scores.append(expert_score)
    return expert_scores

def add_teacher_replay_buffer(teacher_replay_buffer, replay_buffer):
    for (state, action, reward, next_state, done) in replay_buffer.buffer:
        teacher_replay_buffer.push(state, action, reward, next_state, done)

def generate_train_batch(teacher_replay_buffer, replay_buffer,batch_size,expert_ratio=0.25):
    """

    :param teacher_replay_buffer:
    :param replay_buffer:
    :param batch_size:
    :param expert_ratio:  The proportion of expert data
    :return:
    """
    random_batch_list = replay_buffer.sample(int(batch_size * (1 - expert_ratio)))
    expert_batch_list = teacher_replay_buffer.sample(int(batch_size * expert_ratio))
    return {"state_list": np.concatenate((random_batch_list["state_list"],expert_batch_list["state_list"]),axis=0),
            "next_state_list": np.concatenate((random_batch_list["next_state_list"],expert_batch_list["next_state_list"]),axis=0),
            "action_list": np.concatenate((random_batch_list["action_list"],expert_batch_list["action_list"]),axis=0),
            "reward_list": np.concatenate((random_batch_list["reward_list"],expert_batch_list["reward_list"]),axis=0),
            "done_list": np.concatenate((random_batch_list["done_list"],expert_batch_list["done_list"]),axis=0)}

def a_silfd_trainer(args,configs,train_envs,eval_envs):
    envs = train_envs
    args.lamba = configs.ours['lamba']
    redq_td3 = get_redq_td3_agent(args,argparse.Namespace(**configs.td3))
    critic = Critic(args.state_dim, args.action_dim, hidden_width=configs.td3['hidden_width'], num_nets=configs.redq['critic_num']).to(args.device)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=configs.td3['critic_lr'])
    trainning_args = argparse.Namespace(**configs.misc)
    demonstrates_data = DemonstrateDataset(
        file_path=configs.env['demonstrate_path'],
        device=args.device)
    """
    初始化不同的buffer
    """
    replay_buffer = ReplayBuffer(configs.td3['buffer_size'])
    teacher_replay_buffer = ReplayBuffer(configs.td3['buffer_size'])
    episode_replay_buffer = ReplayBuffer(configs.td3['buffer_size'])
    expert_scores = init_teacher_replay_buffer(teacher_replay_buffer,demonstrates_data)
    expert_scores.sort()
    if configs.ours['bc_pre_train']:
       redq_td3.actor.load_state_dict(torch.load(configs.ours['bc_model_path'],
                      map_location=args.device))

    total_steps = 0
    #开始的评估
    evaluate(eval_envs, redq_td3, trainning_args.evaluate_episode, trainning_args.episode_max_steps,total_steps,args.writer)


    evaluate_step = 0
    EPISODE_MAX_REWARD = configs.env['r_max'] * configs.misc['episode_max_steps']
    max_eval_avg_reward = 0
    for episode in range(trainning_args.episodes):
        state = envs.reset()
        done = False
        episode_total_reward = 0
        steps = 0
        while not done and trainning_args.episode_max_steps >= steps:
            if total_steps < configs.td3['start_steps'] :
                action = train_envs.action_space.sample()
            else:
                #Sample actions
                action = (
                        redq_td3.choose_action(state)
                        + np.random.normal(0, redq_td3.max_action * configs.td3['expl_noise'], size=args.action_dim)
                ).clip(-redq_td3.max_action, redq_td3.max_action)
            # Obser reward and next obs
            next_state, reward, done, infos = envs.step(action)
            episode_total_reward += reward
            replay_buffer.push(state, action, reward, next_state, done)
            episode_replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > configs.td3['batch_size']:

                if configs.ours['expert_ratio'] != 0.0:
                    batch_list = generate_train_batch(teacher_replay_buffer, replay_buffer, configs.td3['batch_size'],
                                                      configs.ours['expert_ratio'])
                else:
                    batch_list = replay_buffer.sample(int(configs.td3['batch_size'] * (1 - configs.ours['expert_ratio'])))
                loss_dict = redq_td3.learn(batch_list, critic, critic_target, critic_optimizer,
                                       loss_mode=configs.ours['policy_loss_mode'])
            if configs.ours['lamba_decay'] == True and total_steps%configs.ours['lamba_decay_freq'] == 0 and redq_td3.lamba > configs.ours['lamba_min']  and total_steps > \
                    configs.ours['lamba_decay_start']:
                redq_td3.lamba = lamba_weight_decay_hard(redq_td3.lamba, configs.ours['lamba_decay_rate'])

                if args.writer != None:
                    args.writer.add_scalar("train/lamba", redq_td3.lamba, total_steps)
            if args.render:
                envs.render()
            steps += 1
            total_steps += 1
            evaluate_step +=1
            state = next_state
        """
        每个episode之后
        """
        if total_steps >= configs.misc['num_steps']:
            break
        if episode_total_reward > expert_scores[0] or episode_total_reward > EPISODE_MAX_REWARD / configs.env['max_reward_scale']:
            add_teacher_replay_buffer(teacher_replay_buffer, episode_replay_buffer)
            if args.writer != None:
                args.writer.add_scalar("train/teacher_buffer_size", len(teacher_replay_buffer.buffer), total_steps)
                args.writer.add_scalar("train/teacher_buffer_episode_total_reward", episode_total_reward, total_steps)
            episode_replay_buffer.reset()
            # 最大的长度
            if len(expert_scores) >= configs.ours['teacher_buffer_size'] / trainning_args.episode_max_steps:
                expert_scores.pop(0)
            expert_scores.append(episode_total_reward)
            expert_scores.sort()  # 排序 找到score最大的那个

        if args.writer != None:
            args.writer.add_scalar("train/episode_reward", episode_total_reward, episode)
            args.writer.add_scalar("train/episode_length", steps, episode)
        """
        评估结果
        """
        if evaluate_step >= trainning_args.evaluate_freq_steps :
        # if episode % trainning_args.evaluate_freq == 0:
            print("train/episode:", episode, "reward：", episode_total_reward)
            print("train/episode:", episode, "length：", steps)
            # 评估结果
            average_reward, average_length = evaluate(eval_envs, redq_td3, trainning_args.evaluate_episode,
                                                      trainning_args.episode_max_steps, total_steps, args.writer  )
            evaluate_step = 0

            if max_eval_avg_reward < average_reward and args.save and args.train:
                max_eval_avg_reward = average_reward
                torch.save(redq_td3.state_dict(),
                           args.model_path + f"/asilfd_steps{total_steps}_reward{average_reward:.0f}.pth")

