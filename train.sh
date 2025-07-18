
current_time=$(date +%Y%m%d_%H_%M_%S)
echo $current_time
python3 train.py --save ./workspace/$current_time --seed 666 --config configs_for_develop/driving_bev_sta_config.yaml --gpus 1 --tasks driving_bev_sta
