#!/bin/bash

# 杀掉所有gen2d_ModelResult相关的Python进程
echo "正在查找并杀掉gen2d_ModelResult相关的Python进程..."

# 方法1: 使用pkill命令（推荐）
pkill -f "gen2d_ModelResult_with_odom.py"
pkill -f "gen2d_ModelResult_for_undis.py"
pkill -f "mutli_clip_infer_L4.py"
pkill -f "eval.py"

# 检查是否还有相关进程
remaining_processes=$(pgrep -f "gen2d_ModelResult")
if [ -z "$remaining_processes" ]; then
    echo "所有gen2d_ModelResult相关的Python进程已被成功杀掉"
else
    echo "以下进程仍然在运行:"
    ps -p $remaining_processes -o pid,cmd
    
    # 方法2: 使用kill命令强制杀掉
    echo "正在强制杀掉剩余进程..."
    kill -9 $remaining_processes 2>/dev/null
fi

# 方法3: 使用ps和awk组合（备用方法）
echo "使用备用方法检查..."
ps aux | grep -E "gen2d_ModelResult.*\.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# 最终确认
echo "最终检查..."
ps aux | grep -E "gen2d_ModelResult.*\.py" | grep -v grep
if [ $? -eq 0 ]; then
    echo "警告：仍有相关进程在运行"
else
    echo "确认：所有gen2d_ModelResult相关的Python进程已被清理"
fi