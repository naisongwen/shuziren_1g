###安装ffmpeg
##添加FFmpeg PPA（个人包档案）到你的系统。这个PPA包含了最新版本的FFmpeg和它的依赖。
#以下是在Ubuntu操作系统的命令，其他类型的操作系统，需要更改安装命令

sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update

#安装FFmpeg。
sudo apt-get install ffmpeg

#安装 Miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b

conda config --add channels 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/'

conda create --name py310 python=3.10
source activate py310

#pip install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2 -f https://mirrors.aliyun.com/pytorch-wheels/cu118

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

#以下命令在整合包目录下执行

pip install -r requirements.txt
python webui.py
