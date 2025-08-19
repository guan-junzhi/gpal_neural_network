import os

root_dir = "/data/ai_group/workdirs/gpal_neural_network_group/airflow_workspace"
local_buf = "/data1/remote_workspace_temp"

dirs = {
    "0807_30_epoch": "gpal_neural_network_one_node_traning_job_on_airflow_20250807_12_52_45",
    "0807_test_run": "gpal_neural_network_one_node_traning_job_on_airflow_20250807_08_20_09",
    "0813_zuobin1": "gpal_neural_network_one_node_traning_job_on_airflow_20250813_04_44_17",
    "0813_zuobin2": "gpal_neural_network_one_node_traning_job_on_airflow_20250813_09_45_53"

}

cmd = f"rm -rf {local_buf}/*"
os.system(cmd)

for k in dirs:
    # 复制到本地盘, 实际访问本地盘，网慢的时候用这个，重启更新
    # cmd = f"mkdir -p {os.path.join(local_buf, dirs[k])}"
    # os.system(cmd)
    # cmd = f"cp -r {os.path.join(root_dir, dirs[k], 'log')} {os.path.join(local_buf, dirs[k])}"
    # os.system(cmd)

    # 软链到本地盘, 实际访问网盘，实时更新
    cmd = f"mkdir -p {os.path.join(local_buf)}"
    os.system(cmd)
    cmd = f"ln -s {os.path.join(root_dir, dirs[k], 'log')} {os.path.join(local_buf, k)}"
    os.system(cmd)

cmd = f"ls -l {local_buf}"
os.system(cmd)


# no --bind_all 使用localhost
cmd = f"tensorboard --logdir {local_buf}"
os.system(cmd)

# --bind_all 使用服务器端口
# cmd = f"tensorboard --logdir {local_buf} --bind_all"
# os.system(cmd)
