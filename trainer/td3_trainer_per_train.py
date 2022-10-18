import gym
from trainer.base import get_td3_agent
import numpy as np
import argparse
from utils.trainer_utils import evaluate
import torch
from utils.trainer_utils import DemonstrateDataset
def init_replay_buffer(replay_buffer,demonstrates_data):
    expert_states = demonstrates_data.expert_states
    expert_actions = demonstrates_data.expert_actions
    expert_rewards = demonstrates_data.expert_rewards
    expert_next_states = demonstrates_data.expert_next_states
    expert_dones = demonstrates_data.expert_dones
    for i in range(1,len(expert_states)+1):
        replay_buffer.push(expert_states[i-1],expert_actions[i-1],expert_rewards[i-1],expert_next_states[i-1],expert_dones[i-1])
def td3_pre_train_trainer(args,configs,train_envs,eval_envs):
    envs = train_envs

    td3_agent = get_td3_agent(args,argparse.Namespace(**configs.td3))
    trainning_args = argparse.Namespace(**configs.misc)
    total_steps = 0



    if configs.ours['bc_pre_train']:
       td3_agent.actor.load_state_dict(torch.load(configs.td3['bc_model_path'],
                      map_location=args.device))
    if configs.redq['pretrain_demo']: #用RLfD的方式进行pre train
        demonstrates_data = DemonstrateDataset(
            file_path=configs.env['demonstrate_path'],
            device=args.device)
        init_replay_buffer(td3_agent.replay_buffer,demonstrates_data)
        for i in range(configs.redq['pretrain_epoch']):
            batch_list = td3_agent.replay_buffer.sample(configs.td3['batch_size'])
            loss_dict = td3_agent.learn(batch_list)

    # 开始的评估
    evaluate(eval_envs, td3_agent, trainning_args.evaluate_episode,
                                                 trainning_args.episode_max_steps, total_steps, args.writer)

    max_eval_avg_reward = -2000
    """
    大概135000 step收敛
    """
    evaluate_step = 0
    for episode in range(trainning_args.episodes):
        state = envs.reset()
        done = False
        episode_total_reward = 0
        steps = 0
        while not done and trainning_args.episode_max_steps >= steps:
            if total_steps < configs.td3['start_steps'] :
                action = train_envs.action_space.sample()
            else:
                # Sample actions
                action = (
                        td3_agent.choose_action(state)
                        + np.random.normal(0, td3_agent.max_action * configs.td3['expl_noise'], size=args.action_dim)
                ).clip(-td3_agent.max_action, td3_agent.max_action)
            # Obser reward and next obs
            next_state, reward, done, infos = envs.step(action)
            episode_total_reward += reward
            td3_agent.replay_buffer.push(state, action, reward, next_state, done)
            if len(td3_agent.replay_buffer) > configs.td3['batch_size']:
                batch_list = td3_agent.replay_buffer.sample(configs.td3['batch_size'])
                loss_dict = td3_agent.learn(batch_list)
                # if args.writer != None:
                #     args.writer.add_scalar('update/td_error1', loss_dict['td_error1'], total_steps)
                #     args.writer.add_scalar('update/td_error2', loss_dict['td_error2'], total_steps)
            if args.render:
                envs.render()
            steps += 1
            total_steps += 1
            evaluate_step +=1
            state = next_state
        """
        每个episode之后打印log
        """

        if args.writer != None:
            args.writer.add_scalar("train/episode_reward", episode_total_reward, episode)
            args.writer.add_scalar("train/episode_length", steps, episode)

        """
        评估结果
        """
        # if episode % trainning_args.evaluate_freq == 0:
        if evaluate_step >= trainning_args.evaluate_freq_steps :
            print("train/episode:", episode, "reward：", episode_total_reward)
            print("train/episode:", episode, "length：", steps)
            evaluate_step = 0
            # 评估结果
            average_reward,average_length = evaluate(eval_envs, td3_agent, trainning_args.evaluate_episode,
                                                     trainning_args.episode_max_steps,total_steps,args.writer)

            if max_eval_avg_reward < average_reward and args.save and args.train:
                max_eval_avg_reward = average_reward
                torch.save(td3_agent.state_dict(),
                           args.model_path + f"/td3_steps{total_steps}_reward{average_reward:.0f}.pth")

