# gpal_neural_network

## 1. Introduction

##### directory

    .
    ├── configs_for_develop         # 开发用配置文件
    ├── configs_for_release         # 发布用配置文件（存档）
    ├── gpal_lightning              # 框架/公共组件/基类
    ├── gpal_nn                     # 模型&任务
    │   ├── models                  # 业务模型
    │   │   ├── backbones
    │   │   └── transformers
    │   └── tasks                   # 任务
    │       └── driving_bev_sta     # **静态感知任务**
    ├── initVenv.sh
    ├── readme.md
    ├── README.md
    ├── tools_scripts
    ├── train.py
    └── train.sh

##### progress
-  [multitask_lanenet](http://172.16.2.227:7990/users/jianlongwu/repos/multitask_lanenet/browse)
   -  [x] train
   -  [x] test
   -  [ ] deploy
-  [slotdetect_pointline](http://172.16.2.227:7990/projects/SLOT/repos/slotdetect_pointline/browse)
   -  [ ] train
   -  [ ] test
   -  [ ] deploy
-  [od]()
   -  [ ] train
   -  [ ] test
   -  [ ] deploy

## 2. Train

    bash train.sh

tensorboard

    修改 tools_scripts/remote_tensorboard.py 中感兴趣的对比训练，起一个有意义的名字作为key值
    python tools_scripts/remote_tensorboard.py



## 2. Validation

    bash eval.sh


## 3. Deploy


## 4. Dependence packages

    conda create -n torch20 pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
    pip install pytorch-lightning==2.5.2
    pip install psutil
    pip install scipy
    pip install opencv-python
    pip install tensorboard
    pip install shapely
    pip install lmdb

    部署兼容的环境配置
    conda create -n torch23_deploy python==3.10 pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c 

## 5. Refrence

[pytorch lightning](https://lightning.ai/docs/pytorch/latest/versioning.html#pytorch-support)
    