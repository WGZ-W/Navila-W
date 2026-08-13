
import os
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers.models.siglip.convert_siglip_to_hf import model_name_to_checkpoint

from llava.model.utils import get_model_config

model_name_or_path = "/home/weiguanzhao/navila-siglip-llama3-8b-v1.5-pretrain/llm"

# 在加载前后添加调试信息
print("Loading model...")
print(f"Model path: {model_name_or_path}")

llm = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    # torch_dtype=eval(config.model_dtype),
    device_map="auto"  # 或者 "cuda:0"
)

# 详细检查模型结构
print(f"Model type: {type(llm)}")
print(f"Number of parameter groups: {len(list(llm.parameters()))}")

# 检查每一层的参数
total_params = 0
trainable_params = 0
for name, param in llm.named_parameters():
    layer_params = param.numel()
    total_params += layer_params
    if param.requires_grad:
        trainable_params += layer_params
    print(f"Layer {name}: {layer_params:,} parameters, Trainable: {param.requires_grad}")

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
