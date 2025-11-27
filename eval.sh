#!/bin/bash


current_time=$(date +%Y%m%d_%H_%M_%S)
echo $current_time

echo "[INSTALL ENVS]:"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip install pandas
pip install terminaltables
pip install similaritymeasures
pip install matplotlib
pip install pyquaternion

echo "[READ GLOBAL ENV VAR]:"
airflow_key="gpal_neural_network_one_node_traning_job_on_airflow"
echo "AIRFLOW_CTX_DAG_ID=$AIRFLOW_CTX_DAG_ID"
echo "airflow_key=$airflow_key"
if [[ "$AIRFLOW_CTX_DAG_ID" == "$airflow_key" ]];
then
export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT='/opt/airflow/workdirs/'
export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT"gpal_neural_network_group/airflow_workspace"
export ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT
export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT='/opt/airflow/datasets/'
export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT='/opt/airflow/process-prod-bucket/data_collect'
export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_SSD_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT
export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT='/opt/airflow/local_datasets/'
export ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=1
export ENV_GPAL_NEURAL_NETWORK_WORKSPACE=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT/${AIRFLOW_CTX_DAG_ID}_${current_time}
export ENV_GPAL_NEURAL_NETWORK_GPUS=8

else
export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT='/data/ai_group/workdirs/'
export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT"gpal_neural_network_group/airflow_workspace"
export ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT='workspace/'
export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT='/data/ai_group/datasets/'
export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT='/data/dp_group/process-prod-bucket/data_collect/'
export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_SSD_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT
export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT='/data1/'
export ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=1
export ENV_GPAL_NEURAL_NETWORK_WORKSPACE=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT/$current_time
export ENV_GPAL_NEURAL_NETWORK_GPUS=1
fi

echo ""
echo "[SET LOCAL ENV VAR]:"
echo ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT
echo ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT
echo ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT
echo ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT
echo ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT
echo ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT=$ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT
echo ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE=$ENV_GPAL_NEURAL_NETWORK_WORLD_SIZE
echo ENV_GPAL_NEURAL_NETWORK_WORKSPACE=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE
echo ENV_GPAL_NEURAL_NETWORK_GPUS=$ENV_GPAL_NEURAL_NETWORK_GPUS


echo $1
if [[ $1 == "parking_ipm_sta" ]];
then
    load_from=/home/jovyan/gpal_neural_network/workspace/20251021_09_58_35/checkpoint/epoch=146-step=150000_checkpoint.pth
    config=/home/jovyan/gpal_neural_network/workspace/20251021_09_58_35/config.yaml
elif [[ $1 == "driving_bev_dyn" ]];
then
    tasks=driving_bev_dyn 
    load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/lane_detection_one_node_traning_job_on_airflow_20251011_02_12_25/checkpoint/epoch=5-step=50000_checkpoint.pth
    onnx_path="workspace//20251023_06_55_13_onnx/checkpoint/epoch=5-step=50000_checkpoint_sim.onnx"
    # onnx_path="/data/ai_group/workdirs/od_occ_group/mendeswan/codes/gpal_od_pcdet_calibration_hbm/tools/ptq/model_output/single_frame_moldel_v1_int16_random_calib_data_1021/single_frame_moldel_v1_int16_quantized_model.bc"
    config=configs_for_develop/driving_bev_dyn_config.yaml
    calib_data_save_path="None" 
else
    tasks=driving_bev_sta
    load_from=/data/ai_group/workdirs/gpal_neural_network_group/airflow_workspace/gpal_neural_network_one_node_traning_job_on_airflow_20251124_11_48_13/checkpoint/epoch=0-step=20000_checkpoint.pth
    config=./configs_for_develop/driving_bev_sta_config.yaml
    # onnx_path=/data/ai_group/workdirs/multitask_lanenet_group/tongwang/crosswalk_arrow_6iter_40k/model_int16_quantized_model.bc
    onnx_path=/data/ai_group/workdirs/multitask_lanenet_group/tongwang/crosswalk_arrow_6iter_40k/model_int16_calibrated_model.onnx
    # calib_data_save_path=./tools_scripts/driving_bev_sta/calib_data
    calib_data_save_path="None" 
fi

echo load_from=$load_from
echo config=$config

# python3 eval.py --load_from $load_from --onnx_path $onnx_path  --calib_data_save_path $calib_data_save_path --save $ENV_GPAL_NEURAL_NETWORK_WORKSPACE  --config $config
python3 eval.py --load_from $load_from --save $ENV_GPAL_NEURAL_NETWORK_WORKSPACE  --config $config
