from funasr import AutoModel

path_vad  = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
path_punc = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
vad_model_revision = punc_model_revision = "v2.0.4"

path_asr = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model_revision = "v2.0.4"

model = AutoModel(
            model               = path_asr,
            model_revision      = model_revision,
            vad_model           = path_vad,
            vad_model_revision  = vad_model_revision,
            punc_model          = path_punc,
            punc_model_revision = punc_model_revision,
        )

file_path=''
text = model.generate(input=file_path)[0]["text"]
