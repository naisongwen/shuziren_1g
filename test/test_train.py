import argparse
import psutil
import shutil
import sys,torch
import os
import traceback
import json,yaml
from multiprocessing import cpu_count

path=os.path.join(os.getcwd())
print(path)
sys.path.insert(0,path)

from GPTSoVITS.tools import my_utils
from GPTSoVITS.tools import slice_audio
from GPTSoVITS.tools.my_utils import load_audio, check_for_existance, check_details
from GPTSoVITS.config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix,is_share
from GPTSoVITS.utils import get_hparams_from_file

voice_id='instant_voice_438c74db5c111'
base_dir=f'/data/workdir/voice-clone/{voice_id}'
#文本标注文件
inp_text =f'{base_dir}/output/asr_opt/slicer_opt.list'
#训练集音频文件目录
inp_wav_dir = f'{base_dir}/output/slicer_opt/'
gpu_numbers1a="0"
gpu_numbers1Ba = "0"
gpu_numbers1c ="0"
total_epoch=2
batch_size =8
text_low_lr_rate =0.4
save_every_epoch = 4
if_save_latest  =True
if_save_every_weights  =True
gpu_numbers1Ba = "0"

SoVITS_weight_root="SoVITS_weights"
GPT_weight_root="GPT_weights"

pretrained_sovits_name=["GPTSoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"]
pretrained_gpt_name=["GPTSoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"]

#预训练的中文BERT模型路径
bert_pretrained_dir ="GPTSoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

#预训练的SSL模型路径
cnhubert_base_dir='GPTSoVITS/pretrained_models/chinese-hubert-base'

#预训练的SoVITS-G模型路径
pretrained_s2G=pretrained_sovits_name[0]
#预训练的SoVITS-D模型路径
pretrained_s2D=pretrained_sovits_name[0].replace("s2G","s2D")
#预训练的GPT模型路径
pretrained_s1 = pretrained_gpt_name[0]

with open("GPTSoVITS/configs/s2.json")as f:
    data=f.read()
    data=json.loads(data)
s2_dir="%s/%s"%(exp_root,voice_id)
os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)
if(is_half==False):
    data["train"]["fp16_run"]=False
    batch_size=max(1,batch_size//2)

version="v2"
data["train"]["batch_size"]=batch_size
data["train"]["epochs"]=total_epoch
data["train"]["text_low_lr_rate"]=text_low_lr_rate
data["train"]["pretrained_s2G"]=pretrained_s2G
data["train"]["pretrained_s2D"]=pretrained_s2D
data["train"]["if_save_latest"]=if_save_latest
data["train"]["if_save_every_weights"]=if_save_every_weights
data["train"]["save_every_epoch"]=save_every_epoch
data["train"]["gpu_numbers"]=gpu_numbers1Ba
data["model"]["version"]=version
data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
data["save_weight_dir"]=SoVITS_weight_root
data["name"]=voice_id
data["version"]=version

tmp = os.path.join(s2_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
tmp_config_path="%s/tmp_s2.json"%tmp
with open(tmp_config_path,"w")as f:f.write(json.dumps(data))
from GPTSoVITS import s2_train
from random import randint
print('开始SoVITS训练')
hps=get_hparams_from_file(tmp_config_path)
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(randint(20000, 55555))
os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")
ngpu = torch.cuda.device_count()
s2_train.run(0,ngpu if torch.cuda.is_available() or ngpu != 0 else 1,hps)
