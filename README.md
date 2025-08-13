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
安装一个部署兼容的环境配置（ubuntu22.04 + open_explorer_v3.0.31 + pytorch2.3 + pytorch-lightning==2.3 + cuda11.8）

准备地平线的依赖包（这个路径最好能打包到docker里面固化，下面~/.bashrc会用到）：

    mkdir deploy
    cd deploy
    cp /data/ai_group/workdirs/multitask_lanenet_group/sikong/horizon_j6_open_explorer_v3.0.31-py310_20241231.tar.gz .
    tar -zxvf horizon_j6_open_explorer_v3.0.31-py310_20241231.tar.gz
    cd horizon_j6_open_explorer_v3.0.31-py310_20241231/package/host/
    bash resolve.sh
    tar -zxvf gcc-12.2.0-compiled.tgz

安装基础环境：

    conda create -n torch23_deploy python==3.10 pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    sudo apt-get update
    sudo apt-get -y install graphviz
    sudo apt-get -y install procps
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

解决一个版本冲突问题

    pip install cython==0.29.33

以下concat到~/.bashrc（路径随动于地平线安装包的解压路径）

    export LC_ALL=en_US.utf8
    export LANG=en_US.utf8
    export PATH=/home/jovyan/deploy/horizon_j6_open_explorer_v3.0.31-py310_20241231/package/host/gcc-12.2.0-compiled/bin:$PATH
    export ESTIMATION_ENABLE=1
    export LINARO_GCC_ROOT=/home/jovyan/deploy/horizon_j6_open_explorer_v3.0.31-py310_20241231/package/host/gcc-12.2.0-compiled/bin

安装地平线的包

    bash install.sh

安装pytorch-lightning和其他package

    pip install pytorch-lightning==2.3
    pip install shapely==2.1.1
    pip install loguru
    pip install tensorboardX
    pip install pandas
    pip install terminaltables
    pip install similaritymeasures


装完后可能存在以下incompatible, 集中在horizon-torch-samples和jupter，可能不影响主体业务，先不管

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    horizon-torch-samples 3.0.32 requires fsspec==2022.1.0, but you have fsspec 2025.7.0 which is incompatible.
    horizon-torch-samples 3.0.32 requires Shapely==1.8.0, but you have shapely 2.1.1 which is incompatible.
    horizon-torch-samples 3.0.32 requires torchmetrics==0.5.0, but you have torchmetrics 1.8.1 which is incompatible.
    jupyter-server 2.16.0 requires packaging>=22.0, but you have packaging 21.3 which is incompatible.
    jupyterlab-server 2.27.3 requires requests>=2.31, but you have requests 2.22.0 which is incompatible.


## 5. Refrence

[pytorch lightning](https://lightning.ai/docs/pytorch/latest/versioning.html#pytorch-support)
    