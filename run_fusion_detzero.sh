#!/bin/bash
cd /code
export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT=/data/ai_group/workdirs/
export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=/data/ai_group/workdirs/gpal_neural_network_group/airflow_workspace
export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT=/data/ai_group/datasets/
export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT=/data/dp_group/process-prod-bucket/data_collect/
export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_SSD_ROOT=/data/dp_group/process-prod-bucket/data_collect/
export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT=/data1/
export ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=1
rm -rf workspace/dyn_pth_eval_fusion_detzero_test
mkdir -p workspace/dyn_pth_eval_fusion_detzero_test
python3 eval.py   --load_from ${ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT}/detzero_one_node_traning_job_on_airflow_for_k8s_20260409_14_56_14/checkpoint/epoch=1-step=10000_checkpoint.pth   --save workspace/dyn_pth_eval_fusion_detzero_test   --config configs_for_develop/dyn_fusion_eval_detzero_test.yaml   --gpus 8   > workspace/dyn_pth_eval_fusion_detzero_test/log.txt 2>&1
