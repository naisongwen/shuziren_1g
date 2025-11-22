import torch
if torch.cuda.is_available():
    ngpu = torch.cuda.device_count()
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
    print(gpu_name)
else:
    print('no gpu')
