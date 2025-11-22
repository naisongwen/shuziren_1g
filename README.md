# 文件排布说明
|-app                             #存储视频资源文件夹
|  |-assets
|
|-weights                         #模型权重文件夹
|  |-wav2lip                      #通用wav2lip 模型文件夹(包含onnx和pth)
|  |-syncnet                      #通用专家模型
|  |-yolov8n-face                 #人脸检测模型       
|  |-wflw                         #关键点检测模型
|
|-inference_torch.py              #torch pth推理脚本
|
|-medels                          #网络文件
|  |-wav2lip.py                #通用wav2lip网络结构
|  |-syncnet.py                   #通用专家模型网络结构    
|  |-conv.py                      #网络结构OP相关库
|
|-results                         #视频推理结果文件夹
|-test_data                       #测试音视频素材文件夹
|  |-audios                       #测试音频文件夹
|  |-videos                       #测试视频文件夹
|
|-torchalign                      #torchalign开源库
|-audio.py                        #其他有用脚本——音频处理函数脚本
|-utils.py                        #其他有用脚本——自定义函数脚本
|-hparams.py                      #其他有用脚本——HParams类
|-README.md                       #说明文档


# 安装依赖

## cuda 11.6

## python 库
pip install -r requirements.txt