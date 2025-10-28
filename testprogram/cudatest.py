import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"CUDA 디바이스 이름: {torch.cuda.get_device_name(0)}")