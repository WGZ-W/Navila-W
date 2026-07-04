import json
import os

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from pathlib import Path
from llava.model import *


def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            print(f"GPU {i}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")





model_path = "/mnt/sdc/weiguanzhao/navila-llama3-8b-8f"

config = LlavaLlamaConfig.from_pretrained(model_path, resume=False)
if getattr(config, "resume_path", None) is not None:
    config.resume_path = model_path

model = LlavaLlamaModel(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=4096,

    )


print_gpu_memory()





#
# tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path=str(model_path),  # 明确转换为字符串
#         model_base=None,
#         model_name=get_model_name_from_path(str(model_path))
#     )

# 添加模型参数信息
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila-llama3-8b-8f+vln_mix+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path=str(model_path),  # 明确转换为字符串
#         model_base=None,
#         model_name=get_model_name_from_path(str(model_path))
#     )

# 添加模型参数信息
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")