from pathlib import Path
from typing import Union, List, Dict, Any

import numpy as np
# from typing import Dict, List, Optional, Union
# from pathlib import Path
# import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
# from transformers import LlamaTokenizerFast
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
import os, json
# from model.prismatic import PrismaticVLM
# from model.overwatch import initialize_overwatch
# from model.action_tokenizer import ActionTokenizer
# from model.vision_backbone import DinoSigLIPViTBackbone, DinoSigLIPImageTransform
# from model.llm_backbone import LLaMa2LLMBackbone

import cv2

from llava.dataset import RLDSBatchTransform, RLDSDataset, PaddedCollatorForActionPrediction
from llava.model.action_tokenizer import ActionTokenizer
from llava.model import *
from llava.model.action_tokenizer import ActionTokenizer




# model_path: str = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila-llama3-8b-8f+vln_mix+b2+lr-1e-05+lora-r16+dropout-0.05"
# model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila-llama3-8b-8f+vln_mix+b1+lr-1e-05+lora-r16+dropout-0.05"
# model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila-llama3-8b-8f+vln_mix+b1+lr-0.0005+lora-r32+dropout-0.0"
# model_path = "/mnt/sdc/weiguanzhao/navila-llama3-8b-8f"
model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila-llama3-8b-8f+b2+lr-0.0005+lora-r32+dropout-0.0"
config = LlavaLlamaConfig.from_pretrained(model_path, resume=False)
if getattr(config, "resume_path", None) is not None:
    config.resume_path = model_path

model = LlavaLlamaModel(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=2048,
).to("cuda:0")

# processor = MultiModalProcessor(model)

norm_stats = None
tokenizer = model.tokenizer
action_tokenizer = ActionTokenizer(tokenizer)
image_processor = model.get_vision_tower().image_processor


# 自己训练的 pt 模型导入
dataset_statistics_path = "/mnt/sdc/weiguanzhao/dataset_statistics.json"
if os.path.isfile(dataset_statistics_path):
    with open(dataset_statistics_path, "r") as f:
        norm_stats = json.load(f)
    # policy.norm_stats = norm_stats
    model.norm_stats = norm_stats


batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_processor=image_processor,
    )


data_root_dir = Path("/mnt/sdc/weiguanzhao/OpenFly-rlds")
dataset_name: str = "vln_mix"

vla_dataset = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=tuple([384, 384]),
        shuffle_buffer_size=10_000,
        image_aug=True,
        train=True,
    )

print(f"Dataset length is {len(vla_dataset)}")

collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right"
    )

dataloader = DataLoader(
        vla_dataset,
        batch_size=1,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        pin_memory=False,
    )

# batch = next(iter(dataloader))


for batch_idx, batch in enumerate(dataloader):
    with torch.autocast("cuda"):
        # fmt: off
        generated_ids = model.generate(
            input_ids=batch['input_ids'].to("cuda:0"),  # Shape: [1, seq]
            # pixel_values=pixel_values,  # Shape: [1, 3, res, res] or Dict[str, ...]
            pixel_values=batch['pixel_values'].to("cuda:0"),
            max_new_tokens=8,
            # do_sample=True,
            temperature=0.7,
            bos_token_id=tokenizer.bos_token_id,
        )
    print(f"Generated ids: {generated_ids[:, -8:]}")
    print(f"Labels ids: {batch['labels'][:, -9:-1]}")
    print(f"Input ids: {batch['input_ids']}")

    if (batch_idx + 1) % 100 == 0:
        break


# unnorm_key = "vlnv1"
#
# from llava.model.language_model.llava_llama import LlavaLlamaModel
# action_norm_stats = LlavaLlamaModel.get_action_stats(norm_stats, unnorm_key)
# mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
# action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
# actions = np.where(
#     mask,
#     0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
#     normalized_actions,
# )



# model_name_or_path="IPEC-COMMUNITY/openfly-agent-7b"
# model_name_or_path="/home/weiguanzhao/openfly-agent-7b"
# processor = AutoProcessor.from_pretrained(
#     model_name_or_path,
#     # No fast tokenizer
#     # use_fast=False
# )
# model = AutoModelForVision2Seq.from_pretrained(
#     model_name_or_path,
#     attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
# ).to("cuda:0")

# 添加模型参数信息
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

# dataset_statistics_path = "/home/weiguanzhao/OpenFly2/openfly-agent-7b/dataset_statistics.json"
# if os.path.isfile(dataset_statistics_path):
#     with open(dataset_statistics_path, "r") as f:
#         norm_stats = json.load(f)
#     model.norm_stats = norm_stats
#
image1 = Image.fromarray(cv2.imread("example.png"))
image2 = Image.fromarray(cv2.imread("example2.jpg"))
# text = "Take off, go straight pass the river"
text = "Hello world"

# inputs1 = processor(image1, text, return_tensors="pt")
# inputs2 = processor(image2, text, return_tensors="pt")
# inputs1 = {k: v.to(model.device) for k, v in inputs1.items()}
# inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}


action1 = model.predict_action(image1, text,
                              norm_stats=norm_stats,
                              action_tokenizer=action_tokenizer,
                              unnorm_key="vlnv1")
action2 = model.predict_action(image2, text,
                              norm_stats=norm_stats,
                              action_tokenizer=action_tokenizer,
                              unnorm_key="vlnv1")
# print(prompt)
# inputs = processor(prompt, [image, image, image]).to("cuda:0", dtype=torch.bfloat16)
# # inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
# # action = model.predict_action(**inputs, unnorm_key="vln_norm", do_sample=False)
# action = model.predict_action(**inputs, unnorm_key="vlnv1", do_sample=False)
print(action1)
print(action2)