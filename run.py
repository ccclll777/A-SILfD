import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import numpy as np
import gym
import torch
import datetime
import yaml
import argparse
from tensorboardX import SummaryWriter

from trainer.td3_trainer import td3_trainer
from trainer.redq_td3_trainer import redq_td3_trainer

from trainer.asilfd import a_silfd_trainer
from trainer.td3_trainer_per_train import td3_pre_train_trainer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='mujoco') #mujoco / dmc(deepmind control suit)
    parser.add_argument('--task', type=str, default='ant',choices=['ant','walker2d','hopper','halfcheetah'])
    parser.add_argument('--algo', type=str, default='redq_td3',choices=['A-SILfD','td3','redq_td3'])
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model-path',type=str,default="models")
    parser.add_argument('--save', type=bool, default=True)#是否保存模型
    parser.add_argument('--comet-config', type=str, default="") #configs/comel.yaml
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--gpu-no', default=5, type=int)
    parser.add_argument('--expert-type', default="mix", type=str,choices=['expert','mix','sub'])

    parser.add_argument('--bc-pre-train', default=False, type=bool)

    parser.add_argument('--pretrain_demo', default=False, type=bool)
    """
    调参
    """
    args = parser.parse_known_args()[0]
    print(args)
    """
    根据 benchmark 及任务  确定配置文件路径
    """
    base_dir = os.getcwd()
    configs_path = os.path.join(base_dir,"configs", args.benchmark,args.task+".yaml")
    configs_file = open(configs_path, encoding="utf-8")
    configs = yaml.load(configs_file,Loader=yaml.FullLoader) #配置文件加载
    configs = argparse.Namespace(**configs)
    """
        可能调整配置文件中的参数
    """
    if args.expert_type == 'expert':
        configs.env['demonstrate_path'] = configs.env['demonstrate_path']
    elif args.expert_type == 'mix':
        configs.env['demonstrate_path'] = configs.env['mix_demonstrate_path']
    elif args.expert_type == 'sub':
        configs.env['demonstrate_path'] = configs.env['sub_demonstrate_path']

    args.envs = configs.env['env_name']

    print(configs)
    train_envs = gym.make(configs.env['env_name'])
    eval_envs = gym.make(configs.env['env_name'])

    args.train_envs = train_envs
    args.state_dim = train_envs.observation_space.shape[0]
    args.action_dim = train_envs.action_space.shape[0] or train_envs.action_space.n
    args.max_action = train_envs.action_space.high[0]
    args.min_action = train_envs.action_space.low[0]
    args.action_space = train_envs.action_space
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    eval_envs.seed(args.seed)
    print("envs name", args.envs)
    print("Observations shape:", args.state_dim)
    print("Actions shape:", args.action_dim)
    print("Action range:", np.min(train_envs.action_space.low),
          np.max(train_envs.action_space.high))
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)
        torch.cuda.set_device(args.gpu_no)
    args.writer = None
    args.comel_experiment = None
    #训练模式下才会记录log
    if args.train:
        # log相关
        t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
        args.log_file = f'seed_{args.seed}_{t0}'
        #log path
        log_path = os.path.join(base_dir, "results", args.log_dir, args.benchmark + "_" + args.envs, args.algo, args.log_file)
        print("log_path",log_path)
        #model path
        args.model_path = os.path.join(base_dir, "results", args.model_path, args.benchmark + "_" + args.envs, args.algo, args.log_file)
        print("model_path",args.model_path)
        folder = os.path.exists(args.model_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(args.model_path)
        writer = SummaryWriter(log_path)
        # 写入 yaml 文件
        with open(os.path.join(log_path,args.task+".yaml"), "w") as yaml_file:
            yaml.dump(configs, yaml_file)
        args.writer = writer
    if args.algo == 'td3':
        td3_trainer(args, configs, train_envs, eval_envs)
    elif args.algo == 'redq_td3':
        redq_td3_trainer(args, configs, train_envs, eval_envs)
    elif args.algo == 'A-SILfD':
        a_silfd_trainer(args, configs, train_envs, eval_envs)
    elif args.algo == 'td3_pre_train':
        td3_pre_train_trainer(args, configs, train_envs, eval_envs)
if __name__ == "__main__":
    main()