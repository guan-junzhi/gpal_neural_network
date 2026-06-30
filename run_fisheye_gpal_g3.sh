#!/bin/bash
cd /code
export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT=/data/ai_group/workdirs/
export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=/data/ai_group/workdirs/gpal_neural_network_group/airflow_workspace
export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT=/data/ai_group/datasets/
export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT=/data/dp_group/process-prod-bucket/data_collect/
export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_SSD_ROOT=/data/dp_group/process-prod-bucket/data_collect/
export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT=/data1/
export ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=1
rm -rf workspace/dyn_pth_eval_fisheye_gpal_g3
mkdir -p workspace/dyn_pth_eval_fisheye_gpal_g3
python3 eval.py \
  --load_from /data/ai_group/workdirs/gpal_neural_network_group/airflow_workspace/detzero_one_node_traning_job_on_airflow_for_k8s_20260422_05_29_28/checkpoint/epoch=2-step=16000_checkpoint.pth \
  --save workspace/dyn_pth_eval_fisheye_gpal_g3 \
  --config configs_for_develop/dyn_fsh_eval_gpal_g3.yaml \
  --gpus 8 \
  > workspace/dyn_pth_eval_fisheye_gpal_g3/log.txt 2>&1
