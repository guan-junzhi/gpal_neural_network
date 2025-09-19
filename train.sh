current_time=$(date +%Y%m%d_%H_%M_%S)
echo $current_time

echo "[INSTALL ENVS]:"
sudo apt-get update
sudo apt-get -y install procps
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip install pandas
pip install matplotlib
pip install pyquaternion

echo "[NODE INFO]:"
nvidia-smi 
free -m

echo "[READ GLOBAL ENV VAR]:"
airflow_key="lane_detection_one_node_traning_job_on_airflow"
echo "AIRFLOW_CTX_DAG_ID=$AIRFLOW_CTX_DAG_ID"
echo "airflow_key=$airflow_key"
if [[ "$AIRFLOW_CTX_DAG_ID" == "$airflow_key" ]];
then
export ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT='/opt/airflow/workdirs/'
export ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT"gpal_neural_network_group/airflow_workspace"
export ENV_GPAL_NEURAL_NETWORK_WORKSPACE_ROOT=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT
export ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT='/opt/airflow/datasets/'
export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT='/opt/airflow/process-prod-bucket/data_collect'
export ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_SSD_ROOT='/opt/airflow/process-datasets-bucket/business_datasets/lane_bev_data/'
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


if [[ $1 == "parking_ipm_sta" ]];
then
    tasks=parking_ipm_sta 
    load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20250818_08_19_56_weiwei_ckpt/checkpoint/epoch=4-step=1000_checkpoint_weiwei.pth
    config=configs_for_develop/parking_ipm_sta_config.yaml
elif [[ $1 == "driving_bev_dyn" ]];
then
    tasks=driving_bev_dyn 
    # load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20250911_12_22_22/checkpoint/epoch=16-step=30000_checkpoint.pth
    load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/lane_detection_one_node_traning_job_on_airflow_20250919_02_18_35/checkpoint/epoch=9-step=10000_checkpoint.pth
    config=configs_for_develop/driving_bev_dyn_config.yaml
elif [[ $1 == "radar4d_nn_sdk" ]];
then
    tasks=radar4d_nn_sdk 
    load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20250818_08_19_56_weiwei_ckpt/checkpoint/epoch=4-step=1000_checkpoint_weiwei.pth
    config=configs_for_develop/radar4d_nn_sdk_config.yaml
else
    tasks=driving_bev_sta 
    load_from=$ENV_GPAL_NEURAL_NETWORK_AIRFLOW_WORKSPACE_ROOT/gpal_neural_network_one_node_traning_job_on_airflow_20250813_09_45_53/checkpoint/ep006_wangtong.pth
    config=configs_for_develop/driving_bev_sta_config.yaml

fi

echo tasks=$tasks
echo load_from=$load_from
echo config=$config

echo "[AIRFLOW ARGS CHECK]:"

if [[ -n "${TRAIN_ARGS:-}" ]]; then
    echo TRAIN_ARGS=$TRAIN_ARGS
    echo "[RUN WITH TRAIN_ARGS]"
    eval python3 train.py $TRAIN_ARGS
    exit $?
fi

# python3 train.py --save $ENV_GPAL_NEURAL_NETWORK_WORKSPACE --seed 304 --config $config --gpus $ENV_GPAL_NEURAL_NETWORK_GPUS --tasks $tasks
# load_from
python3 train.py --load_from $load_from --save $ENV_GPAL_NEURAL_NETWORK_WORKSPACE --seed 304 --config $config --gpus $ENV_GPAL_NEURAL_NETWORK_GPUS --tasks $tasks
