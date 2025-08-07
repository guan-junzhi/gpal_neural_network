current_time=$(date +%Y%m%d_%H_%M_%S)
echo $current_time

echo "[INSTALL ENVS]:"
sudo apt-get update
sudo apt-get install procps
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip install pandas

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
export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT='/opt/airflow/local_datasets/'
export WORLD_SIZE=1
worspace=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT/${AIRFLOW_CTX_DAG_ID}_${current_time}
gpus=8

else
export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT='/data/ai_group/workdirs/'
export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT"gpal_neural_network_group/airflow_workspace"
export ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT='workspace/'
export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT='/data/ai_group/datasets/'
export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT='/data/dp_group/process-prod-bucket/data_collect/'
export ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT='/data1/'
export WORLD_SIZE=1
worspace=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT/$current_time
gpus=2
fi
echo ""
echo "[SET LOCAL ENV VAR]:"
echo ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT
echo ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT
echo ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT
echo ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT
echo ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT=$ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT
echo ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT=$ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT
echo WORLD_SIZE=$WORLD_SIZE
echo worspace=$worspace

load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20250728_13_04_38_2epoch_ckpt/checkpoint/epoch=1-step=3500_checkpoint_wangtong.pth

echo load_from=$load_from
echo gpus=$gpus

# python3 train.py --save ./workspace/$current_time --seed 666 --config configs_for_develop/driving_bev_sta_config.yaml --gpus 2 --tasks driving_bev_sta

# python3 train.py --load_from workspace/20250728_03_57_54/checkpoint/epoch=1-step=2000_checkpoint.pth --save ./workspace/$current_time --seed 666 --config configs_for_develop/driving_bev_sta_config.yaml --gpus 2 --tasks driving_bev_sta
python3 train.py --load_from $load_from --save $worspace --seed 666 --config configs_for_develop/driving_bev_sta_config.yaml --gpus $gpus --tasks driving_bev_sta



# python3 train.py --resume ./workspace/20250718_19_35_25_load/checkpoint/epoch=1-step=50_checkpoint.pth --seed 666 --config configs_for_develop/driving_bev_sta_config.yaml --gpus 1 --tasks driving_bev_sta
# python3 train.py --resume ./workspace/20250718_19_35_25_load/checkpoint/epoch=1-step=50_checkpoint.pth  --config ./workspace/20250718_19_35_25_load/config.yaml --save ./workspace/$current_time --gpus 1 --tasks driving_bev_sta
