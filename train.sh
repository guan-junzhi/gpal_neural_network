#!/usr/bin/env bash
set -euo pipefail

current_time=$(date +%Y%m%d_%H_%M_%S)
echo "$current_time"

# ------------------------------------------------------------
# 0) 先激活 conda 环境，并强制后续 pip/python 用同一个解释器
# ------------------------------------------------------------
echo "[INIT CONDA]:"
source /opt/conda/etc/profile.d/conda.sh
conda activate torch23_deploy
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "[INSTALL ENVS]:"
apt-get update
apt-get -y install procps
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install pandas
python -m pip install matplotlib
python -m pip install pyquaternion
python -m pip install onnxruntime
python -m pip install tensorboard
python -m pip install psutil
python -m pip install lmdb




PY_BIN="$(python -c 'import sys; print(sys.executable)')"
echo "[INFO] python=$PY_BIN"
python -V

# ------------------------------------------------------------
# 1) Horizon OpenExplorer 环境注入（解决 horizon_plugin_pytorch）
#    你给的 export 需要放在这里，并补齐 PYTHONPATH/LD_LIBRARY_PATH
# ------------------------------------------------------------
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export LC_CTYPE=C.UTF-8
export PYTHONIOENCODING=utf-8

export HORIZON_OE=/workspace/deploy/horizon_j6_open_explorer_v3.0.31-py310_20241231
export PATH=$HORIZON_OE/package/host/gcc-12.2.0-compiled/bin:$PATH
export ESTIMATION_ENABLE=1
export LINARO_GCC_ROOT=$HORIZON_OE/package/host/gcc-12.2.0-compiled

export PYTHONPATH="${PYTHONPATH:-}"

if [ -d "$HORIZON_OE/package/host/python3.10/lib/python3.10/site-packages" ]; then
  export PYTHONPATH="$HORIZON_OE/package/host/python3.10/lib/python3.10/site-packages:$PYTHONPATH"
fi
if [ -d "$HORIZON_OE/package/host/python/lib/python3.10/site-packages" ]; then
  export PYTHONPATH="$HORIZON_OE/package/host/python/lib/python3.10/site-packages:$PYTHONPATH"
fi

echo "[INFO] PYTHONPATH=${PYTHONPATH}"

# 关键：补齐动态库搜索路径（具体 lib 路径可按实际再加）
export LD_LIBRARY_PATH="$LINARO_GCC_ROOT/lib64:$LINARO_GCC_ROOT/lib:${LD_LIBRARY_PATH:-}"

echo "[INFO] PYTHONPATH=$PYTHONPATH"
echo "[INFO] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# ------------------------------------------------------------
# 2) 依赖安装：必须用 python -m pip，别用裸 pip / python3
#    （建议后续移到镜像里固化；目前先让它跑通）
# ------------------------------------------------------------
echo "[INSTALL ENVS]:"
apt-get update
apt-get -y install procps

python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ || true
python -m pip install -U --no-cache-dir pandas matplotlib pyquaternion onnxruntime tensorboard psutil

echo "[IMPORT CHECK]:"
python -c "import psutil; print('psutil ok')"
python -c "import horizon_plugin_pytorch; print('horizon_plugin_pytorch ok:', horizon_plugin_pytorch.__file__)" || \
  (echo '[ERROR] horizon_plugin_pytorch not found. Check HORIZON_OE path + PYTHONPATH.' && exit 2)

echo "[NODE INFO]:"
nvidia-smi || true
free -m || true

echo "[READ GLOBAL ENV VAR]:"
airflow_key="lane_detection_one_node_traning_job_on_airflow"
echo "AIRFLOW_CTX_DAG_ID=$AIRFLOW_CTX_DAG_ID"
echo "airflow_key=$airflow_key"

if [[ "${TASK_IN_AIRFLOW:-0}" == "1" ]];
then
  export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT='/opt/airflow/workdirs/'
  export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT="${ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT}gpal_neural_network_group/airflow_workspace"
  export ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT="$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT"
  export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT='/opt/airflow/datasets/'
  export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT='/opt/airflow/process-prod-bucket/data_collect'
  export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_SSD_ROOT="$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT"
  export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT='/opt/airflow/local_datasets/'
  export ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=1
  export ENV_GPAL_NEURAL_NETWORK_WORKSPACE="${ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT}/${AIRFLOW_CTX_DAG_ID}_${current_time}"
  export ENV_GPAL_NEURAL_NETWORK_GPUS="$DAG_PARAM_GPUS"

  tasks="$DAG_PARAM_TASKS"
  load_from="$DAG_PARAM_LOAD_FROM"
  config="$DAG_PARAM_CONFIG"
  seed="$DAG_PARAM_SEED"
  gpus="$DAG_PARAM_GPUS"

else
  export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT='/data/ai_group/workdirs/'
  export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT="${ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT}gpal_neural_network_group/airflow_workspace"
  export ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT='workspace/'
  export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT='/data/ai_group/datasets/'
  export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT='/data/dp_group/process-prod-bucket/data_collect/'
  export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_SSD_ROOT="$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT"
  export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT='/data1/'
  export ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=1
  export ENV_GPAL_NEURAL_NETWORK_WORKSPACE="${ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT}/$current_time"
  export ENV_GPAL_NEURAL_NETWORK_GPUS=1

  if [[ "${1:-}" == "parking_ipm_sta" ]];
  then
      tasks=parking_ipm_sta
      load_from=/data/ai_group/workdirs/gpal_neural_network_group/zww/slot_pretrained/best.pth
      config=configs_for_develop/parking_ipm_sta_config.yaml
  elif [[ "${1:-}" == "driving_bev_dyn" ]];
  then
      tasks=driving_bev_dyn
      load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/lane_detection_one_node_traning_job_on_airflow_20250924_15_00_27/checkpoint/epoch=2-step=18000_checkpoint.pth
      config=configs_for_develop/driving_bev_dyn_config.yaml
  elif [[ "${1:-}" == "radar4d_nn_sdk" ]];
  then
      tasks=radar4d_nn_sdk
      load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20250818_08_19_56_weiwei_ckpt/checkpoint/epoch=4-step=1000_checkpoint_weiwei.pth
      config=configs_for_develop/radar4d_nn_sdk_config.yaml
  else
      tasks=driving_bev_sta
      load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20251215_12_44_00/checkpoint/epoch=0-step=60000_checkpoint.pth
      config=configs_for_develop/driving_bev_sta_config.yaml
  fi
  seed=304
fi

echo ""
echo "[SET LOCAL ENV VAR]:"
echo ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT="$ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT"
echo ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT="$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT"
echo ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT="$ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT"
echo ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT="$ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT"
echo ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT="$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT"
echo ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT="$ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT"
echo ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE="$ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE"
echo ENV_GPAL_NEURAL_NETWORK_WORKSPACE="$ENV_GPAL_NEURAL_NETWORK_WORKSPACE"
echo ENV_GPAL_NEURAL_NETWORK_GPUS="$ENV_GPAL_NEURAL_NETWORK_GPUS"

echo "tasks=$tasks"
echo "load_from=$load_from"
echo "config=$config"
echo "seed=$seed"

# ------------------------------------------------------------
# 3) 运行训练：用 python（不要 python3），确保同一解释器
# ------------------------------------------------------------
if [[ "$load_from" == "None" ]];
then
  python train.py --save "$ENV_GPAL_NEURAL_NETWORK_WORKSPACE" --seed "$seed" --config "$config" --gpus "$ENV_GPAL_NEURAL_NETWORK_GPUS" --tasks "$tasks"
else
  python train.py --load_from "$load_from" --save "$ENV_GPAL_NEURAL_NETWORK_WORKSPACE" --seed "$seed" --config "$config" --gpus "$ENV_GPAL_NEURAL_NETWORK_GPUS" --tasks "$tasks"
fi

##########################################################################

# current_time=$(date +%Y%m%d_%H_%M_%S)
# echo $current_time

# echo "[INSTALL ENVS]:"
# apt-get update
# apt-get -y install procps
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
# pip install pandas
# pip install matplotlib
# pip install pyquaternion
# pip install onnxruntime
# pip install tensorboard
# pip install psutil


# echo "[NODE INFO]:"
# nvidia-smi 
# free -m

# echo "[READ GLOBAL ENV VAR]:"
# airflow_key="lane_detection_one_node_traning_job_on_airflow"
# echo "AIRFLOW_CTX_DAG_ID=$AIRFLOW_CTX_DAG_ID"
# echo "airflow_key=$airflow_key"
# if [[ "$TASK_IN_AIRFLOW" == "1" ]];
# then
# export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT='/opt/airflow/workdirs/'
# export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT"gpal_neural_network_group/airflow_workspace"
# export ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT
# export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT='/opt/airflow/datasets/'
# export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT='/opt/airflow/process-prod-bucket/data_collect'
# export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_SSD_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT
# export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT='/opt/airflow/local_datasets/'
# export ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=1
# export ENV_GPAL_NEURAL_NETWORK_WORKSPACE=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT/${AIRFLOW_CTX_DAG_ID}_${current_time}
# export ENV_GPAL_NEURAL_NETWORK_GPUS=$DAG_PARAM_GPUS

# tasks=$DAG_PARAM_TASKS 
# load_from=$DAG_PARAM_LOAD_FROM
# config=$DAG_PARAM_CONFIG
# seed=$DAG_PARAM_SEED
# gpus=$DAG_PARAM_GPUS

# else
# export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT='/data/ai_group/workdirs/'
# export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT"gpal_neural_network_group/airflow_workspace"
# export ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT='workspace/'
# export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT='/data/ai_group/datasets/'
# export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT='/data/dp_group/process-prod-bucket/data_collect/'
# export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_SSD_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT
# export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT='/data1/'
# export ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=1
# export ENV_GPAL_NEURAL_NETWORK_WORKSPACE=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT/$current_time
# export ENV_GPAL_NEURAL_NETWORK_GPUS=1

# if [[ $1 == "parking_ipm_sta" ]];
# then
#     tasks=parking_ipm_sta 
#     load_from=/data/ai_group/workdirs/gpal_neural_network_group/zww/slot_pretrained/best.pth
#     config=configs_for_develop/parking_ipm_sta_config.yaml
# elif [[ $1 == "driving_bev_dyn" ]];
# then
#     tasks=driving_bev_dyn 
#     # load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20250911_12_22_22/checkpoint/epoch=16-step=30000_checkpoint.pth
#     load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/lane_detection_one_node_traning_job_on_airflow_20250924_15_00_27/checkpoint/epoch=2-step=18000_checkpoint.pth
#     config=configs_for_develop/driving_bev_dyn_config.yaml
# elif [[ $1 == "radar4d_nn_sdk" ]];
# then
#     tasks=radar4d_nn_sdk 
#     load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20250818_08_19_56_weiwei_ckpt/checkpoint/epoch=4-step=1000_checkpoint_weiwei.pth
#     config=configs_for_develop/radar4d_nn_sdk_config.yaml
# else
#     tasks=driving_bev_sta 
#     load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20251215_12_44_00/checkpoint/epoch=0-step=60000_checkpoint.pth
#     config=configs_for_develop/driving_bev_sta_config.yaml

# fi
# seed=304
# fi

# echo ""
# echo "[SET LOCAL ENV VAR]:"
# echo ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT
# echo ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT
# echo ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT
# echo ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT
# echo ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT
# echo ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT=$ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT
# echo ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=$ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE
# echo ENV_GPAL_NEURAL_NETWORK_WORKSPACE=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE
# echo ENV_GPAL_NEURAL_NETWORK_GPUS=$ENV_GPAL_NEURAL_NETWORK_GPUS

# echo tasks=$tasks
# echo load_from=$load_from
# echo config=$config
# echo seed=$seed

# if [[ "$load_from" == "None" ]];
# then
# python3 train.py --save $ENV_GPAL_NEURAL_NETWORK_WORKSPACE --seed $seed --config $config --gpus $ENV_GPAL_NEURAL_NETWORK_GPUS --tasks $tasks
# else
# python3 train.py --load_from $load_from --save $ENV_GPAL_NEURAL_NETWORK_WORKSPACE --seed $seed --config $config --gpus $ENV_GPAL_NEURAL_NETWORK_GPUS --tasks $tasks
# fi
