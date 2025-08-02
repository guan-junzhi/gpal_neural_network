
current_time=$(date +%Y%m%d_%H_%M_%S)
echo $current_time
# python3 train.py --save ./workspace/$current_time --seed 666 --config configs_for_develop/driving_bev_sta_config.yaml --gpus 2 --tasks driving_bev_sta

# python3 train.py --load_from workspace/20250728_03_57_54/checkpoint/epoch=1-step=2000_checkpoint.pth --save ./workspace/$current_time --seed 666 --config configs_for_develop/driving_bev_sta_config.yaml --gpus 2 --tasks driving_bev_sta
python3 train.py --load_from workspace/20250728_13_04_38_2epoch_ckpt/checkpoint/epoch=1-step=3500_checkpoint_wangtong.pth --save ./workspace/$current_time --seed 666 --config configs_for_develop/driving_bev_sta_config.yaml --gpus 1 --tasks driving_bev_sta



# python3 train.py --resume ./workspace/20250718_19_35_25_load/checkpoint/epoch=1-step=50_checkpoint.pth --seed 666 --config configs_for_develop/driving_bev_sta_config.yaml --gpus 1 --tasks driving_bev_sta
# python3 train.py --resume ./workspace/20250718_19_35_25_load/checkpoint/epoch=1-step=50_checkpoint.pth  --config ./workspace/20250718_19_35_25_load/config.yaml --save ./workspace/$current_time --gpus 1 --tasks driving_bev_sta
