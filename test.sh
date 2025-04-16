#!/bin/bash
cd build
cmake ..
make
# 定义两个可执行文件的路径
APP1="/home/mayh/MYH/Project_code/C++/bhr8-fc2-mujoco/build/bin/main_app"
APP2="/home/mayh/MYH/Library/bitbot/bitbot-frontend"

# 检查第一个应用是否存在并执行
if [ -f "$APP1" ]; then
    echo "正在启动 main_app..."
    gnome-terminal --tab --title="BHR8-FC2-Mujoco" -- bash -c "$APP1; exec bash"
else
    echo "错误：未找到 $APP1"
    exit 1
fi

# 检查第二个应用是否存在并执行
if [ -f "$APP2" ]; then
    echo "正在启动 bitbot-frontend..."
    gnome-terminal --tab --title="BitBot Frontend" -- bash -c "$APP2; exec bash"
else
    echo "错误：未找到 $APP2"
    exit 1
fi

echo "两个应用已成功启动！"

