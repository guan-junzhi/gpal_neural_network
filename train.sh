#!/usr/bin/env bash
set -euo pipefail

current_time="$(date +%Y%m%d_%H_%M_%S)"
echo "$current_time"

# ------------------------------------------------------------
# 0) Conda + 禁止 user site 污染
# ------------------------------------------------------------
echo "[INIT CONDA]:"
if [[ "${TASK_IN_AIRFLOW:-0}" == "1" ]];
then
source /opt/conda/etc/profile.d/conda.sh
conda activate torch23_deploy

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

PY_BIN="$(python -c 'import sys; print(sys.executable)')"
echo "[INFO] python=$PY_BIN"
python -V

# ------------------------------------------------------------
# 1) 基础依赖（尽量少装；能固化到镜像更好）
# ------------------------------------------------------------
echo "[INSTALL OS PKGS]:"
apt-get update
apt-get -y install --no-install-recommends procps
rm -rf /var/lib/apt/lists/*

echo "[INSTALL PY PKGS]:"
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ || true
python -m pip install --no-cache-dir -U psutil lmdb tensorboard pandas matplotlib pyquaternion onnxruntime opencv-python-headless

# ------------------------------------------------------------
# 2) Horizon OpenExplorer + horizon_plugin_pytorch（关键：用 wheel 安装）
# ------------------------------------------------------------
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export LC_CTYPE=C.UTF-8
export PYTHONIOENCODING=utf-8

# 你当前的 OE 路径（必须在 Pod 内真实存在）
export HORIZON_OE="${HORIZON_OE:-/workspace/deploy/horizon_j6_open_explorer_v3.0.31-py310_20241231}"
echo "[INFO] HORIZON_OE=$HORIZON_OE"

# 可选：工具链 PATH（如果你的代码/插件运行时需要）
export LINARO_GCC_ROOT="$HORIZON_OE/package/host/gcc-12.2.0-compiled"
if [ -d "$LINARO_GCC_ROOT/bin" ]; then
  export PATH="$LINARO_GCC_ROOT/bin:$PATH"
fi
export LD_LIBRARY_PATH="$LINARO_GCC_ROOT/lib64:$LINARO_GCC_ROOT/lib:${LD_LIBRARY_PATH:-}"

# 关键：安装 horizon_plugin_pytorch wheel 到 conda env
WHL="$HORIZON_OE/package/host/ai_toolchain/horizon_plugin_pytorch-2.5.9+cu118.torch230-cp310-cp310-linux_x86_64.whl"
if [ -f "$WHL" ]; then
  echo "[INFO] Installing horizon_plugin_pytorch from wheel: $WHL"
  python -m pip install --no-cache-dir -U "$WHL"
else
  echo "[ERROR] horizon_plugin_pytorch wheel not found: $WHL"
  echo "[HINT] Ensure HORIZON_OE is available in this pod (baked in image or mounted via volume)."
  exit 2
fi

# 如果 OE 还提供额外的 python 包目录（可选补充；不是主路径）
# 注意：保持禁用 user-site，避免 /root/.local 污染
OE_SP1="$HORIZON_OE/package/host/python3.10/lib/python3.10/site-packages"
OE_SP2="$HORIZON_OE/package/host/python/lib/python3.10/site-packages"
export PYTHONPATH=""
if [ -d "$OE_SP1" ]; then PYTHONPATH="$OE_SP1:$PYTHONPATH"; fi
if [ -d "$OE_SP2" ]; then PYTHONPATH="$OE_SP2:$PYTHONPATH"; fi
export PYTHONPATH="${PYTHONPATH#:}"  # 去掉可能的前导冒号
echo "[INFO] PYTHONPATH=$PYTHONPATH"
echo "[INFO] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "[IMPORT CHECK]:"
python -c "import psutil; print('psutil ok')"
python -c "import lmdb; print('lmdb ok')"
python -c "import horizon_plugin_pytorch; import os; print('horizon_plugin_pytorch ok:', horizon_plugin_pytorch.__file__)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cudnn:', torch.backends.cudnn.version(), 'torch_file:', torch.__file__)"

# ------------------------------------------------------------
# 3) 运行环境变量（Airflow / local 兼容）
# ------------------------------------------------------------
echo "[NODE INFO]:"
nvidia-smi || true
free -m || true

fi

echo "[READ GLOBAL ENV VAR]:"
airflow_key="${airflow_key:-lane_detection_one_node_traning_job_on_airflow}"

# 在 set -u 下，确保 AIRFLOW_CTX_* 不为空
export AIRFLOW_CTX_DAG_ID="${AIRFLOW_CTX_DAG_ID:-unknown_dag}"
export AIRFLOW_CTX_TASK_ID="${AIRFLOW_CTX_TASK_ID:-unknown_task}"
export AIRFLOW_CTX_RUN_ID="${AIRFLOW_CTX_RUN_ID:-unknown_run}"
export AIRFLOW_CTX_TRY_NUMBER="${AIRFLOW_CTX_TRY_NUMBER:-0}"
export AIRFLOW_CTX_EXECUTION_DATE="${AIRFLOW_CTX_EXECUTION_DATE:-unknown_ts}"

echo "AIRFLOW_CTX_DAG_ID=$AIRFLOW_CTX_DAG_ID"
echo "airflow_key=$airflow_key"

if [[ "${TASK_IN_AIRFLOW:-0}" == "1" ]]; then
  export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT='/opt/airflow/workdirs/'
  export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT="${ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT}gpal_neural_network_group/airflow_workspace"
  export ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT="$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT"
  export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT='/opt/airflow/datasets/'
  export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT='/opt/airflow/process-prod-bucket/data_collect'
  export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_SSD_ROOT="$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT"
  export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT='/opt/airflow/local_datasets/'
  export ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=1
  export ENV_GPAL_NEURAL_NETWORK_WORKSPACE="${ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT}/${AIRFLOW_CTX_DAG_ID}_${current_time}"
  export ENV_GPAL_NEURAL_NETWORK_GPUS="${DAG_PARAM_GPUS:-8}"

  tasks="${DAG_PARAM_TASKS:-None}"
  load_from="${DAG_PARAM_LOAD_FROM:-None}"
  config="${DAG_PARAM_CONFIG:-None}"
  seed="${DAG_PARAM_SEED:-304}"
else
  export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT='/data/ai_group/workdirs/'
  export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT="${ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT}gpal_neural_network_group/airflow_workspace"
  export ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT='workspace/'
  export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT='/data/ai_group/datasets/'
  export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT='/data/dp_group/process-prod-bucket/data_collect/'
  export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_SSD_ROOT="$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT"
  export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT='/data1/'
  export ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=1
  export ENV_GPAL_NEURAL_NETWORK_WORKSPACE="${ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT}/$current_time"
  export ENV_GPAL_NEURAL_NETWORK_GPUS=1

  # 本地模式参数保持你原逻辑
  if [[ "${1:-}" == "parking_ipm_sta" ]]; then
    tasks=(parking_ipm_sta)
    load_from=/data/ai_group/workdirs/gpal_neural_network_group/zww/slot_pretrained/best.pth
    config=configs_for_develop/parking_ipm_sta_config.yaml
  elif [[ "${1:-}" == "driving_bev_dyn" ]]; then
    tasks=(driving_bev_dyn)
    load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/detzero_one_node_traning_job_on_airflow_for_k8s_20260315_12_03_50/checkpoint/epoch=4-step=40000_checkpoint.pth
    config=configs_for_develop/driving_bev_dyn_config.yaml
  elif [[ "${1:-}" == "radar4d_nn_sdk" ]]; then
    tasks=(radar4d_nn_sdk)
    load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20250818_08_19_56_weiwei_ckpt/checkpoint/epoch=4-step=1000_checkpoint_weiwei.pth
    config=configs_for_develop/radar4d_nn_sdk_config.yaml
  elif [[ "${1:-}" == "driving_bev_sta_dyn" ]]; then
    tasks=(driving_bev_dyn driving_bev_sta)
    load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/detzero_one_node_traning_job_on_airflow_for_k8s_20260315_12_03_50/checkpoint/epoch=4-step=40000_checkpoint.pth
    config=configs_for_develop/driving_bev_dyn_config_muti_with_sta.yaml
  else
    tasks=(driving_bev_sta)
    load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20251215_12_44_00/checkpoint/epoch=0-step=60000_checkpoint.pth
    config=configs_for_develop/driving_bev_sta_config.yaml
  fi
  seed=304
fi

echo ""
echo "[SET LOCAL ENV VAR]:"
echo "ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT"
echo "ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT"
echo "ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT"
echo "ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT"
echo "ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT"
echo "ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT=$ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT"
echo "ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=$ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE"
echo "ENV_GPAL_NEURAL_NETWORK_WORKSPACE=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE"
echo "ENV_GPAL_NEURAL_NETWORK_GPUS=$ENV_GPAL_NEURAL_NETWORK_GPUS"

echo "tasks=$tasks"
echo "load_from=$load_from"
echo "config=$config"
echo "seed=$seed"

# ------------------------------------------------------------
# 4) 启动训练（同一解释器）
# ------------------------------------------------------------
echo "[START TRAINING]:"
if [[ "$load_from" == "None" ]]; then
  python train.py --save "$ENV_GPAL_NEURAL_NETWORK_WORKSPACE" --seed "$seed" --config "$config" --gpus "$ENV_GPAL_NEURAL_NETWORK_GPUS" --tasks ""${tasks[@]}""
else
  python train.py --load_from "$load_from" --save "$ENV_GPAL_NEURAL_NETWORK_WORKSPACE" --seed "$seed" --config "$config" --gpus "$ENV_GPAL_NEURAL_NETWORK_GPUS" --tasks ""${tasks[@]}""
fi



