import json
import os

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.model.action_tokenizer import ActionTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "/home/weiguanzhao/navila-llama3-8b-8f"  # 或者完整的HuggingFace路径
model_path = "/home/weiguanzhao/navila-llama3-8b-8f"

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaConfig
import torch

# 检查模型目录内容
if os.path.exists(model_path):
    files = os.listdir(model_path)
    print("模型目录中的文件:", files)


with open(os.path.join(model_path, "config.json"), "r") as f:
    config_data = json.load(f)

# 创建配置对象
config = LlavaConfig.from_dict(config_data)

try:
    # 使用 LLaVA 专用加载器
    processor = LlavaProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("使用 LLaVA 专用加载器成功")
except Exception as e:
    print(f"LLaVA 加载失败: {e}")

