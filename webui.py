# -*- coding: utf-8 -*-

import gradio as gr
from subprocess import Popen
import torch
from multiprocessing import cpu_count
import netifaces
import time
import sys
import traceback
import argparse
import yaml
import psutil
import os
import traceback
from glob import glob

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import string

now_dir = os.getcwd()
sys.path.insert(0, now_dir)
parent_directory = os.path.dirname(os.path.abspath(__file__))
path=os.path.join(parent_directory,'GPTSoVITS')
sys.path.append(path)
sys.path.append(".")

import request_utils
from GPTSoVITS.tools.i18n.i18n import I18nAuto
from GPTSoVITS.tools.asr.config import asr_dict

import process_avatar2
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix,is_share

from GPTSoVITS import inference

def print_caller_info():
    print('loading webui...')
    for frame in traceback.extract_stack():
        print(f"  File: {frame.filename}, Line: {frame.lineno}, Function: {frame.name}")
print_caller_info()

def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

i18n = I18nAuto(language=os.environ.get('language','Auto'))
def load_yaml_file(file_path):
    with open(file_path, 'r',encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

import requests

config=load_yaml_file('config.yaml')
access_token=config['access_token']
data_dir=config['data_dir']
avatar_dir=f'{data_dir}/video_upload'
voice_dir=f'{data_dir}/voice-clone'


print('声音克隆保存路径:'+voice_dir)
print('形象克隆保存路径:'+avatar_dir)
os.environ['voice_dir']=voice_dir

import train_gpt_sovits

headers = {
    "Accept": "application/json",
    "Authorization": "Bearer "+ access_token
}


os.makedirs(avatar_dir,exist_ok=True)
os.makedirs(voice_dir,exist_ok=True)
os.makedirs(f'{data_dir}/videos/',exist_ok=True)
os.makedirs(f'{data_dir}/audios/',exist_ok=True)
os.makedirs(f'{data_dir}/texts/',exist_ok=True)
os.makedirs(f'{data_dir}/generated_audios/',exist_ok=True)
os.makedirs(f'{data_dir}/results/',exist_ok=True)
os.makedirs(f'{data_dir}/video_upload',exist_ok=True)
os.makedirs(f'{data_dir}/voice_clone',exist_ok=True)
os.makedirs('temp/',exist_ok=True)

allowed_paths=[f'{data_dir}/generated_audios']

voice_map={
        "Alloy(男，OpenAI)":"en-US-AlloyMultilingualNeural",
        "Echo（男，OpenAI）":"en-US-EchoMultilingualNeural ",
}

n_cpu=cpu_count()

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# 判断是否有能用来训练和加速推理的N卡
ok_gpu_keywords={"10","16","20","30","40","A2","A3","A4","P4","A50","500","A60","70","80","90","M4","T4","TITAN","L4","4060","H"}
set_gpu_numbers=set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper()for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(i).total_memory/ 1024/ 1024/ 1024+ 0.4))

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = ("%s\t%s" % ("0", "CPU"))
    gpu_infos.append("%s\t%s" % ("0", "CPU"))
    set_gpu_numbers.add(0)
    default_batch_size = int(psutil.virtual_memory().total/ 1024 / 1024 / 1024 / 2)
gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers=str(sorted(list(set_gpu_numbers))[0])

def fix_gpu_number(input):#将越界的number强制改到界内
    try:
        if(int(input)not in set_gpu_numbers):return default_gpu_numbers
    except:return input
    return input

def fix_gpu_numbers(inputs):
    output=[]
    try:
        for input in inputs.split(","):output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs

def filter_filelist(dir_name,types):
    filtered_files = []
    files = os.listdir(dir_name)
    for file in files:
        filename, extension = os.path.splitext(file)
        if extension in types:
            filtered_files.append(file)
    return filtered_files

def get_voices():
    voice_list=[]
    voice_pattern=os.path.join(voice_dir,'*')
    file_list = glob(voice_pattern)
    for file in file_list:
        if os.path.isdir(file):
            voice_list.append(os.path.basename(file))
    return voice_list

def update_voices():
    voices=get_voices()
    return {"choices": voices,"__type__": "update"}

voices=get_voices()
def avatar_list():
    from glob import glob
    avatars =[]
    avatar_pattern=os.path.join(avatar_dir,'*')
    avatar_info_file_list = glob(avatar_pattern)
    for avatar_info_file in avatar_info_file_list:
        if os.path.isdir(os.path.join(avatar_dir,avatar_info_file)):
            avatar_name, ext = os.path.splitext(os.path.basename(avatar_info_file))
            avatars.append(avatar_name)
    return avatars

avatars=avatar_list()

def update_avatars():
    avatars=avatar_list()
    return {"choices": avatars,"__type__": "update"}

p_training=None
def start_train(voice_id,audio_path,asr_model,asr_size,asr_lang,asr_precision,batch_size,sovits_epcoh,gpt_epoch):
    global p_training
    if p_training:
        yield "训练进行中"
        return
    if not audio_path:
        yield "请上传音频"
        return

    record_id=-1
    status="COMPLETED"

    p_training=True
    yield "训练进行中"
    s="训练完成,请查看控制台训练日志"
    if not voice_id:
        voice_id = generate_random_string(10)
    try:
        os.environ['voice_dir'] = voice_dir
        import importlib
        importlib.reload(train_gpt_sovits)
        train_gpt_sovits.main(voice_id,audio_path,asr_model,asr_size,asr_lang,asr_precision,batch_size,sovits_epcoh,gpt_epoch)
        p_training=None
    except Exception as e:
        p_training=None
        status="FAILED"
        print(e)
        print(traceback.format_exc())
        s="训练失败，请看控制台日志"
    
    data={
        "id":record_id,
        "status":status
    }
    yield s

def sovits_weights(voice_id):
    SoVITS_weight_root=f"{voice_dir}/{voice_id}/SoVITS_weights"
    SoVITS_names = ["s2G2333k.pth"]
    if os.path.exists(SoVITS_weight_root):
        for name in os.listdir(SoVITS_weight_root):
            if name.endswith(".pth"):SoVITS_names.append(name)
    return SoVITS_names

def gpt_weights(voice_id):
    GPT_names = ["s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"]
    GPT_weight_root=f'{voice_dir}/{voice_id}/GPT_weights'
    if os.path.exists(GPT_weight_root):
        for name in os.listdir(GPT_weight_root):
            if name.endswith(".ckpt"): GPT_names.append(name)
    return GPT_names

def update_model(voice_id):
    s=sovits_weights(voice_id)
    g=gpt_weights(voice_id)
    return {"choices": s,"__type__": "update","value":s[0]},{"choices": g,"__type__": "update","value":g[0]}

def confirm_voice(voice_id,sovits_model,gpt_model):
    SoVITS_weight_root=f"{voice_dir}/{voice_id}/SoVITS_weights"
    for name in os.listdir(SoVITS_weight_root):
        if sovits_model not in name:os.remove(os.path.join(SoVITS_weight_root,name))

    GPT_weight_root=f'{voice_dir}/{voice_id}/GPT_weights'
    for name in os.listdir(GPT_weight_root):
        if gpt_model not in name:os.remove(os.path.join(GPT_weight_root,name))
    return '模型已保存'

def audio_listen(voice_id,text,sovits,gpt):
    output_path=f"{data_dir}/generated_audios/{voice_id}_{str(int(time.time()))}.wav"
    #inference.synthesize2(model_dir,text,i18n('粤语'),output_path)
    model_dir=f"{voice_dir}/{voice_id}/"
    output_path=f"{data_dir}/generated_audios/{voice_id}_{str(int(time.time()))}.wav"
    SoVITS_weight=f"{voice_dir}/{voice_id}/SoVITS_weights/{sovits}"
    if "s2G2333k.pth" in sovits:
        SoVITS_weight="GPTSoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"

    GPT_weight=f'{voice_dir}/{voice_id}/GPT_weights/{gpt}'
    if "s1bert25hz" in gpt:
        GPT_weight="GPTSoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    inference.synthesize2(model_dir,text,i18n('多语种混合'),output_path,SoVITS_weight,GPT_weight)
    return output_path

def audio_gen(voice_id,text):
    output_path=f"{data_dir}/generated_audios/{voice_id}_{str(int(time.time()))}.wav"
    #inference.synthesize2(model_dir,text,i18n('粤语'),output_path)
    model_dir=f"{voice_dir}/{voice_id}/"
    output_path=f"{data_dir}/generated_audios/{voice_id}_{str(int(time.time()))}.wav"

    inference.synthesize2(model_dir,text,i18n('多语种混合'),output_path)
    return output_path

def video_gen0(video_path,audio_path,pads=0,result_path=None):
    audio_name=os.path.basename(audio_path)
    audio_name, extension = os.path.splitext(audio_name)
    video_name=os.path.basename(video_path)
    avatar_name, extension = os.path.splitext(video_name)
    avatar_path=f'{avatar_dir}/{avatar_name}'
    coords_paths=glob(os.path.join(avatar_path, '*.pkl'))
    if len(coords_paths)==0:
        process_avatar2.main(video_path,avatar_name,avatar_path,pads)

    if not result_path:
        result_path=f'{data_dir}/results/{avatar_name}_{audio_name}.mp4'
    import inference_fast2
    inference_fast2.main(avatar_name,avatar_path,audio_path,pads,result_path)

p_inference=None
def video_gen(video_path,text_path,video_dropdown,audio_path,pads,audio_ouput_info):
    global p_inference
    
    if(p_inference!=None):
        print('视频正在合成中')
        yield f'推理进行中...'
        return
    audio_path=audio_path if audio_path else audio_ouput_info
    if not audio_path:
        print('未提供音频，或者试听没有成功')
        yield '未提供音频，或者试听没有成功'
        return

    print(audio_path)
    audio_name=os.path.basename(audio_path)
    audio_name, extension = os.path.splitext(audio_name)
    
    try:
        result_video_path=None
        video_path = video_path if video_path else text_path
        print(video_path)
        if os.path.exists(video_path):
            status='COMPLETED'
            p_inference=True
            video_name=os.path.basename(video_path)
            avatar_name, extension = os.path.splitext(video_name)
            result_video_path=f'{data_dir}/results/{avatar_name}_{audio_name}.mp4'
            video_gen0(video_path,audio_path,pads,result_video_path)
            p_inference=None

        elif video_dropdown:
            avatar_name=video_dropdown
            p_inference=True
            result_video_path=f'{data_dir}/results/{avatar_name}_{audio_name}.mp4'
            yield f'正在合成视频...'
            import inference_fast2
            avatar_path=f'{avatar_dir}/{avatar_name}'
            inference_fast2.main(avatar_name,avatar_path,audio_path,pads,result_video_path)
            
            status='COMPLETED'
            p_inference=None
     
        else:
            print('未提供视频,请等待上传完成或者确保输入的视频地址有效')
            yield '未提供视频,请等待上传完成或者确保输入的视频地址有效'
            return

        print(f'合成已完成,已保存在{result_video_path}')
        yield f'合成已完成,已保存在{result_video_path}'
    except Exception as e:
        print(e)
        p_inference=None
        yield f'合成出错，请看控制台详情'

p_batch=None
def batch_process(voice_id,drive_mode,align_mode):
    status='COMPLETED'
    global p_batch
    if p_batch!=None:
        print('批处理进行中，结果保存在results目录')
        yield '批处理进行中，结果保存在results目录'
        return
    
    if drive_mode=='音频驱动数字人':
        audio_dir=os.path.join(data_dir,'audios')
    else:
        audio_dir=os.path.join(data_dir,'generated_audios')
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)

        text_file_dir=os.path.join(data_dir,'texts')
        txt_files = filter_filelist(text_file_dir,['.txt'])
        print(f'{len(txt_files)}个文本待合成...')
        yield f'{len(txt_files)}个文本待合成...'
        for txt_file in txt_files:
            input_text_file=os.path.join(data_dir,'texts',txt_file)
            with open(input_text_file, 'r', encoding="utf-8") as file:
                txt=file.read()
                print(f'正在合成文本：{txt[0:100 if len(txt)>100 else len(txt)]}')
                audio_gen(voice_id,target_languag,text)
        
        print('文本合成完毕')
    try:
        import copy
        p_batch=True
        audio_files = filter_filelist(audio_dir,['.mp3','.wav','.MP3','.WAV'])
        video_files = filter_filelist(os.path.join(data_dir,'videos'),['.mp4','.MP4'])
        audio_len=len(audio_files)
        video_len=len(video_files)
        if align_mode =='按照数量少的对齐':
            if audio_len<video_len:
                video_files=video_files[0:audio_len]
            else:
                audio_files=audio_files[0:video_len]
        elif align_mode =='按照数量多的对齐':
            if audio_len<video_len:
                new_audio_files=copy.copy(audio_files)
                while(len(new_audio_files)!=video_len):
                    if len(audio_files)==0:
                        break;
                    for audio in audio_files:
                        new_audio_files.append(audio)
                        if len(new_audio_files)==video_len:
                            break
                audio_files=new_audio_files
            else:
                new_video_files=copy.copy(video_files)
                while(len(new_video_files)!=audio_len):
                    if len(video_files)==0:
                        break;
                    for video in video_files:
                        new_video_files.append(video)
                        if len(new_video_files)==audio_len:
                            break
                video_files=new_video_files
        else:
            pass
        
        print(f'{video_len}视频，{audio_len}音频待合成')
        yield f'{video_len}视频，{audio_len}音频待合成'
        if align_mode =='全重复（笛卡尔）对齐':
            # 遍历文件列表，删除文件
            for audio_name in audio_files:
                audio_path=os.path.join(audio_dir,audio_name)
                for video_name in video_files:
                    video_name_without_extension = os.path.splitext(os.path.basename(video_name))[0]
                    video_path = os.path.join(data_dir,'videos',video_name)
                    result_video_path=f'{data_dir}/results/{video_name_without_extension}_{audio_name.replace(".","_")}.mp4'
                    print(f'"{audio_name}-"{video_name}"正在合成...')
                    video_gen0(video_path,audio_path)
                    print(f'"{video_name}"-"{audio_name}" 合成已完成')
        else:
            for index, video_name in enumerate(video_files):
                audio_name=audio_files[index]
                video_path = os.path.join(data_dir,'videos',video_name)
                audio_path=os.path.join(audio_dir,audio_name)
                print(f'"{video_name}"-"{audio_name}"正在合成...')
                video_gen0(video_path,audio_path)
                print(f'"{video_name}"-"{audio_name}"合成完毕')
        
        p_batch=None
        yield "批处理完成"
    except Exception as e:
        status='FAILED'
        print(e)
        p_batch=None
        yield '批处理出错,请看控制台详细日志输出'

def on_drive_mode_change(drive_mode):
    if drive_mode == '音频驱动数字人':
        return "请将音频文件（wav，mp3格式）放在运行包的data>audios文件夹中",{"__type__": "update","visible": False,"visible": False}
    else:
        return "请将文本文件（txt格式）放在运行包的data>texts文件夹中",{"__type__": "update","visible": True,"visible": True}

def order(package):
    if package=='基础会员1个月':
        packageName='LOCAL_BASIC_MONTH_1'
    elif package=='基础会员3个月':
        packageName='LOCAL_BASIC_MONTH_3'
    elif package=='基础会员1年':
        packageName='LOCAL_BASIC_MONTH_12'
    elif package=='尊享会员1个月':
        packageName='LOCAL_VIP_MONTH_1'
    elif package=='尊享会员3个月':
        packageName='LOCAL_VIP_MONTH_3'
    elif package=='尊享会员1年':
        packageName='LOCAL_VIP_MONTH_12'
    else:
        packageName='LOCAL_FREE_3_DAYS'
    
    try:
        import qrcode
        from PIL import Image
        from io import BytesIO
        qr_image  = qrcode.make(codeUrl)
        img_byte_arr = BytesIO()
        qr_image.save(img_byte_arr, format='PNG')
        image = Image.open(img_byte_arr)
        image = image.resize((300, 300))
        return "请用微信扫描下方二维码支付",image
    except Exception as e:
        print(e)
        return "下单异常，请联系客服（wx:373055922）",None

port=7860

with gr.Blocks(title="一站式数字人生成系统") as app:
    gr.Markdown(value=config['copyright'])
    with gr.Tabs():
        with gr.TabItem("声音复刻"):
            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem("声音训练"):
                        with gr.Row():
                            gr.Markdown(
                                value="建议1分钟以上干净语音素材，不要有杂音，音频时长越长，训练轮数越大")
                        with gr.Row():
                            input_audio_name = gr.Textbox(label="克隆声音名称")
                            input_audio_path = gr.Audio(label="请上传音频",type='filepath')

                        with gr.Row():
                            batch_size = gr.Slider(minimum=1,maximum=40,step=1,label=i18n("每张显卡的处理批次大小"),value=default_batch_size,interactive=True)
                            sovits_total_epoch = gr.Slider(minimum=1,maximum=25,step=1,label="SoVITS 总训练轮数，不建议太高",value=8,interactive=True)
                            gpt_total_epoch = gr.Slider(minimum=2,maximum=50,step=1,label="GPT 总训练轮数",value=15,interactive=True)
                        
                        with gr.Row():
                            asr_model = gr.Dropdown(
                                        label       = i18n("语音模型"),
                                        choices     = list(asr_dict.keys()),
                                        interactive = True,
                                        value="达摩 ASR (中文)"
                                    )
                            asr_size = gr.Dropdown(
                                        label       = i18n("模型尺寸"),
                                        choices     = ["large"],
                                        interactive = True,
                                        value="large"
                                    )
                            asr_lang = gr.Dropdown(
                                        label       = i18n("语言设置"),
                                        choices     = ["zh","yue"],
                                        interactive = True,
                                        value="zh"
                                    )
                            asr_precision = gr.Dropdown(
                                        label       = i18n("数据精度"),
                                        choices     = ["float32"],
                                        interactive = True,
                                        value="float32"
                                    )
                            def change_lang_choices(key): #根据选择的模型修改可选的语言
                                # return gr.Dropdown(choices=asr_dict[key]['lang'])
                                return {"__type__": "update", "choices": asr_dict[key]['lang'],"value":asr_dict[key]['lang'][0]}
                            def change_size_choices(key): # 根据选择的模型修改可选的模型尺寸
                                # return gr.Dropdown(choices=asr_dict[key]['size'])
                                return {"__type__": "update", "choices": asr_dict[key]['size'],"value":asr_dict[key]['size'][-1]}
                            def change_precision_choices(key): #根据选择的模型修改可选的语言
                                if key =="Faster Whisper (多语种)":
                                    if default_batch_size <= 4:
                                        precision = 'int8'
                                    elif is_half:
                                        precision = 'float16'
                                    else:
                                        precision = 'float32'
                                else:
                                    precision = 'float32'
                                # return gr.Dropdown(choices=asr_dict[key]['precision'])
                                return {"__type__": "update", "choices": asr_dict[key]['precision'],"value":precision}
                            asr_model.change(change_lang_choices, [asr_model], [asr_lang])
                            asr_model.change(change_size_choices, [asr_model], [asr_size])
                            asr_model.change(change_precision_choices, [asr_model], [asr_precision])

                        with gr.Row():
                            button_train = gr.Button("开启训练", variant="primary",visible=True)
                            train_info=gr.Textbox(label="训练进程输出信息")
                        with gr.Row():
                            gr.Markdown("注：首次运行选择不同语音处理模型，会自动下载相应模型，可能需要一些时间，特别是多语种模型，需要开启VPN，后续运行该软件不需要再次开启。")
                    button_train.click(start_train,[input_audio_name,input_audio_path,asr_model,asr_size,asr_lang,asr_precision,batch_size,sovits_total_epoch,gpt_total_epoch],[train_info])
                with gr.Tabs():
                    with gr.TabItem("声音试听"):
                        with gr.Row():
                            voice_dropdown = gr.Dropdown(label="选择声音",choices=voices)
                            refresh_voice_button = gr.Button("刷新声音", variant="primary",interactive=True)
                        with gr.Row():
                            sovits_dropdown = gr.Dropdown(label="选择SoVITS模型")
                            gpt_dropdown = gr.Dropdown(label="选择GPT模型")
                            refresh_model_button = gr.Button("刷新模型", variant="primary",interactive=True)
                        with gr.Row():
                            input_text = gr.Textbox(label="请输入文案",lines=5)
                        with gr.Row():
                            button_gen_audio = gr.Button("试听", variant="primary",visible=True)
                            audio_ouput_info = gr.Audio(label=i18n("输出的语音"),type='filepath')
                        with gr.Row():
                            button_fix_model = gr.Button("就选此模型", variant="primary",visible=True)
                            model_save_info=gr.Textbox(label="输出信息")
                button_gen_audio.click(audio_listen,[voice_dropdown,input_text,sovits_dropdown,gpt_dropdown],[audio_ouput_info])
                refresh_voice_button.click(fn=update_voices,inputs=[],outputs=[voice_dropdown])
                voice_dropdown.change(fn=update_model,inputs=[voice_dropdown],outputs=[sovits_dropdown,gpt_dropdown])
                refresh_model_button.click(fn=update_model,inputs=[voice_dropdown],outputs=[sovits_dropdown,gpt_dropdown])
                button_fix_model.click(confirm_voice,[voice_dropdown,sovits_dropdown,gpt_dropdown],[model_save_info])

        with gr.TabItem(i18n("视频合成")):
            with gr.Row():
                gr.Markdown(
            value="建议提供3分钟以内的不说话人物正面形象视频，人脸偏向幅度不宜过大，注意每一帧都要有人脸。如果上传较慢，也可以输入本地视频路径。")
            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem("请上传视频"):
                        video_path = gr.Video(label="请上传视频(mp4格式)")
                    with gr.TabItem("填写视频路径"):
                        text_path=gr.Textbox(label="请输入本地视频路径(mp4格式)",placeholder='c:\\xxx.mp4')
                    with gr.TabItem("选择现有视频"):
                        with gr.Row():
                            video_dropdown= gr.Dropdown(label="视频列表",choices=avatars,interactive=True)
                            refresh_avatar_button = gr.Button("刷新",variant="primary",interactive=True)
                    refresh_avatar_button.click(fn=update_avatars,inputs=[],outputs=[video_dropdown])
                with gr.Tabs():
                    with gr.TabItem("请上传音频"):
                        with gr.Row():
                            audio_path = gr.Audio(label="请上传音频",type='filepath')
                    with gr.TabItem("文字生成语音"):
                        with gr.Row():
                            voice_dropdown = gr.Dropdown(label="声音列表",choices=voices)
                            refresh_voice_button = gr.Button("刷新", variant="primary",interactive=True)
                            #language_dropdown = gr.Dropdown(label="语言列表",choices=voices)
                        with gr.Row():
                            input_text = gr.Textbox(label="请输入文案",lines=5)
                        #with gr.Row():
                        #    target_language=gr.Radio(label='选择目标语言',choices=["中文", "英文", "日文", "粤语", "韩文", "多语种混合(粤语)"],value='中文')
                        with gr.Row():
                            button_gen_audio = gr.Button("生成音频", variant="primary",visible=True)
                            audio_ouput_info = gr.Audio(label=i18n("输出的语音"),type='filepath')
                refresh_voice_button.click(fn=update_voices,inputs=[],outputs=[voice_dropdown])
            with gr.Row():
                pads=gr.Slider(label="调节口部区域", minimum=0, maximum=3, value=0,step=1, interactive=True)
            with gr.Row():
                button_gen_video = gr.Button("视频合成", variant="primary",visible=True)
                video_ouput_info=gr.Textbox(label="视频生成输出信息")
            
        button_gen_audio.click(audio_gen,[voice_dropdown,input_text],[audio_ouput_info])
        button_gen_video.click(video_gen,[video_path,text_path,video_dropdown,audio_path,pads,audio_ouput_info],[video_ouput_info])
        with gr.TabItem(i18n("视频批量合成")):
            gr.Markdown(
            value="本功能将自动将videos目录中的视频跟audios 目录中的音频进行配对合成数字人口播视频。如果选择文本驱动，系统会将videos目录中的视频和texts 中的文本配对合成视频。文本会自动先生成音频。texts目录中的每个文本包含一条文案，每条文案在200-300字符以内。合成后的视频存储在results 目录中。上述目录都会自动创建在/data/workdir/下。")
            gr.Markdown("""### 第一步 选择驱动数字人方式!""")
            with gr.Row():
                drive_mode=gr.Radio(['音频驱动数字人','文本驱动数字人'],value='音频驱动数字人')
            with gr.Row():
                md=gr.Markdown("请将音频文件（wav，mp3格式）放在运行包的audios文件夹中")
                voice_dropdown = gr.Dropdown(label="声音列表",choices=voices, visible=False)
                #target_language=gr.Radio(label='选择目标语言',choices=["中文", "英文", "日文", "粤语", "韩文", "多语种混合(粤语)"],value='中文', visible=False)
            drive_mode.change(on_drive_mode_change, drive_mode,[md,voice_dropdown])

            gr.Markdown("""### 第二步 选择视频和音频（文本）数量对齐方式!""")
            with gr.Row():
                align_mode=gr.Radio(['按照数量少的对齐','按照数量多的对齐','全重复（笛卡尔）对齐'],value='按照数量少的对齐')
            with gr.Row():
                submit_btn=gr.Button("提交合成", variant="primary",visible=True)
                batch_ouput_info=gr.Textbox(label="批处理输出信息")

        submit_btn.click(batch_process,[voice_dropdown,drive_mode,align_mode],[batch_ouput_info])
        
        if config['display_payinfo']==True:
            with gr.TabItem(i18n("服务价格")):
                with gr.Row():
                    import gradio as gr
                    import pandas as pd
                    member_level=['免费会员3天','基础会员1年','尊享会员1年']
                    data={"会员等级":member_level,
                            '声音克隆数量':['5','不限','不限'],
                            '形象克隆数量':['5','不限','不限'],
                            '视频合成次数':['5','不限','不限'],
                            '批处理合成':['无','无','不限'],
                            '智能文案':['无','无','500条'],
                            '原价（元）':['0','999','1499'],
                            '限时折扣价（元）':['0','199','399'],
                            }
                    df = pd.DataFrame(data)
                    gr.DataFrame(value=df)
                with gr.Row():
                    package=gr.Radio(label='请选择套餐服务',choices=member_level[1:],value='基础会员1年')
                with gr.Row():
                    gr.Markdown(value="请用微信扫描右边二维码支付.如需对公转账，请联系客服微信：373055922")
                    qr_image2=gr.Image(scale=3)
                    gr.Markdown(value="请用微信扫描左边二维码支付，如需对公转账，请联系客服微信：373055922")
                with gr.Row():
                    buy_btn=gr.Button("下单", variant="primary",visible=True)
                    order_ouput_info=gr.Textbox(label="下单输出信息")
                buy_btn.click(order,package,[order_ouput_info,qr_image2])
    
    app.queue(max_size=10).launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        quiet=True,
        allowed_paths=allowed_paths
        #server_port=port
    )
