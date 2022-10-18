# Accelerating Self-Imitation Learning from Demonstrations

##### Code for Accelerating Self-Imitation Learning from Demonstrations via Policy Constraints and Q-Ensemble.

# 1.Installation

To run experiments, you will need to install the following packages preferably in a conda virtual environment

- gym==0.18.0
- mujoco_py
- torch==1.11.0
- tqdm==4.64.0
- tensorboardX==2.5.1
- scipy==1.8.0
- numpy==1.22.3

Suggested build environment:

~~~shell
```
conda create -n a-silfd python=3.8
conda activate a-silfd
pip install -r requirements.txt
```
~~~

# 2.How to run the code

To run the code with the default parameters, simply execute the following command:

## 2.1 A-SILfD

### 2.1.1 Expert Demonstrations

Run A-SILfD on environment Ant-v2, Hopper-v2, Walker2d-v2, HalfCheetah-v2, running seed = 10:

```shell
python run.py --task=ant --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert
python run.py --task=hopper --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert
python run.py --task=walker2d --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert
python run.py --task=halfcheetah --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert
```

### 2.1.2  Mixed Expert Demonstrations 

```shell
python run.py --task=ant --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=mix
python run.py --task=hopper --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=mix
python run.py --task=walker2d --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=mix
python run.py --task=halfcheetah --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=mix
```



### 2.1.3 Sub-optimal Expert Demonstrations

```shell
python run.py --task=ant --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=sub
python run.py --task=hopper --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=sub
python run.py --task=walker2d --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=sub
python run.py --task=halfcheetah --algo=A-SILfD --seed=10 --save=True --train=True --gpu-no=0 --expert-type=sub
```

## 2.2 REDQ-TD3

### 2.2.1 REDQ-TD3

```shell
python run.py --task=ant --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 
python run.py --task=hopper --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 
python run.py --task=walker2d --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 
python run.py --task=halfcheetah --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 
```



### 2.2.2 REDQ-TD3-BC

```shell
python run.py --task=ant --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert --bc-pre-train=True
python run.py --task=hopper --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert --bc-pre-train=True
python run.py --task=walker2d --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert --bc-pre-train=True
python run.py --task=halfcheetah --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert --bc-pre-train=True
```

### 2.2.3 REDQ-TD3-LfD

```shell
python run.py --task=ant --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert --pretrain_demo=True
python run.py --task=hopper --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert --pretrain_demo=True
python run.py --task=walker2d --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert --pretrain_demo=True
python run.py --task=halfcheetah --algo=redq_td3 --seed=10 --save=True --train=True --gpu-no=0 --expert-type=expert --pretrain_demo=True
```

## The other Baseline code implementations are as follows：



| Algorithm | code                                              |
| --------- | ------------------------------------------------- |
| AWAC      | https://github.com/ikostrikov/jaxrl               |
| IQL       | https://github.com/ikostrikov/implicit_q_learning |
| SAIL      | https://github.com/illidanlab/SAIL                |
| OPOLO     | https://github.com/illidanlab/opolo-code          |
| DAC       | https://github.com/illidanlab/opolo-code          |

