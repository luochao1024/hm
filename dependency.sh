#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install -y libxrender-dev libsm6 libxext6 zlib1g-dev libsnappy-dev python3-pip python3-dev python3-opengl
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
pip3 install gym numpy scipy cmake
pip3 install gym['atari']
git clone https://github.com/greydanus/baby-a3c.git

# git clone https://github.com/luochao1024/hm.git