#!/bin/bash
# Copyright (c) 2024 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.


dataset_path=$1
run_type=$2
version=v3.0.31
container_name=$(whoami)_OE_v3.0.400
host_name=$(echo "3.0.31" |awk -F "." '{ print $1"-"$2"-"$3 }')

if [ -z "$dataset_path" ];then
  echo "Please specify the dataset path"
  exit
fi
dataset_path=$(readlink -f "$dataset_path")

echo "Docker version is ${version}"
echo "Dataset path is $(readlink -f "$dataset_path")"

open_explorer_path=$(readlink -f "$(dirname "$0")")
echo "OpenExplorer package path is $open_explorer_path"

if [ "$run_type" == "cpu" ];then
    echo "Start Docker container in CPU mode."
    docker run -it --rm \
      --network host \
      --hostname "OE-J6-CPU-$host_name" \
      --name $container_name \
      -v "$open_explorer_path":/open_explorer \
      -v "$dataset_path":/data/horizon_j6/data \
      -e OE_FTP_SERVER -e OE_FTP_USER_INFO \
      openexplorer/ai_toolchain_ubuntu_22_j6_cpu:"$version"
else
    echo "Start Docker container in GPU mode."
    docker run -it --rm \
      --network host \
      --hostname "OE-J6-GPU-$host_name" \
      --name $container_name \
      --gpus all \
      --shm-size="15g" \
      -v "$open_explorer_path":/open_explorer \
      -v "$dataset_path":/data/horizon_j6/data \
      -v /data:/data \
      -e OE_FTP_SERVER -e OE_FTP_USER_INFO \
      oe_v3.0.31_od_occ_bugfix_compiler:latest
    #   openexplorer/ai_toolchain_ubuntu_22_j6_gpu:"$version"
      
fi
    #   -v /data1/gjz/j6/compile_model/od_occ_py_infer_onnx_framework/od_occ_model/data/id4_share:/open_explorer/od_occ_model/data/id4_share1 \
