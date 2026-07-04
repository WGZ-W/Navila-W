import torch
import torch.nn as nn
try:
   model = nn.Conv2d(3, 64, kernel_size=3)
   input_data = torch.randn(1, 3, 224, 224, device='cuda')
   output = model(input_data)
except RuntimeError as e:
   if 'CUDNN_STATUS_NOT_INITIALIZED' in str(e):
       print("cuDNN 初始化失败，请检查显卡驱动和内存！")
   else:
       raise e