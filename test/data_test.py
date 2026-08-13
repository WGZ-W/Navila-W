import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from llava.model.action_tokenizer import ActionTokenizer
from llava.dataset import Dataset, RLDSBatchTransform, RLDSDataset, PaddedCollatorForActionPrediction
from llava.model import *
import torch


vln_mix = [("vlnv" + str(idx), 1.0) for idx in range(1, 21)]

model_path = "/mnt/sdc/weiguanzhao/navila-llama3-8b-8f"

local_rank = 0
device = torch.device(f"cuda:{local_rank}")

config = LlavaLlamaConfig.from_pretrained(model_path, resume=False)
config.model_dtype = torch.bfloat16
config.model_dtype = config.model_dtype.__str__()
if getattr(config, "resume_path", None) is not None:
    config.resume_path = model_path

model = LlavaLlamaModel(
    config=config,
    attn_implementation="flash_attention_2",
    # model_max_length=2048,
    model_max_length=4096,
).to(device)

# 自己训练的 pt 模型导入
dataset_statistics_path = "/mnt/sdc/weiguanzhao/dataset_statistics.json"
if os.path.isfile(dataset_statistics_path):
    with open(dataset_statistics_path, "r") as f:
        norm_stats = json.load(f)
    # policy.norm_stats = norm_stats
    model.norm_stats = norm_stats

tokenizer = model.tokenizer
action_tokenizer = ActionTokenizer(tokenizer)
image_processor = model.get_vision_tower().image_processor

unnorm_key = "vlnv1"

batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_processor=image_processor,
    )

data_root_dir = Path("/mnt/sdc/weiguanzhao/OpenFly-rlds-my")
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

sum_wrong = 0
# 测试数据集
for batch_idx, batch in enumerate(dataloader):
    for k, v in batch.items():
        print(f"Key {k}'s value is : {v.shape}")

    # truth_label = batch["labels"][:, -9:]
    # wrong_label = truth_label <= 127000
    # wrong_sum = wrong_label.sum()
    # print(f"Batch {batch_idx} has {wrong_sum} wrong labels")
    # if wrong_sum > 0:
    #     sum_wrong += 1
    # print(f"batch_idx{batch_idx}'s label is : {batch['labels'][:, -10:]}")
    # print(batch["labels"])
# print(sum_wrong)
# action = torch.tensor([127744, 127999, 127999, 127999, 127872, 127872, 127872, 127872]) # 0
# action = torch.tensor([127999, 127744, 127999, 127999, 127872, 127872, 127872, 127872]) # 9
# action = torch.tensor([127999, 127999, 127999, 127744, 127872, 127872, 127872, 127872]) # 3
# action = torch.tensor([127999, 127999, 127744, 127999, 127872, 127872, 127872, 127872]) # 2
# action = torch.tensor([127999, 127914, 127999, 127999, 127872, 127872, 127872, 127872]) # 1
# action = torch.tensor([127999, 127999, 127999, 127999, 127999, 127999, 127999, 127999]) # 1
# action = torch.tensor([127999, 127872, 127872, 127872, 127872, 524, 82, 29])
# action = torch.tensor([127744, 127999, 127872, 127872, 127872,
#          127872,    524,     82])
# normalized_actions = action_tokenizer.decode_token_ids_to_actions(action.cpu().numpy())
# action_norm_stats = model.get_action_stats(norm_stats, unnorm_key)
# mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
# action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
# actions = np.where(
#     mask,
#     0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
#     normalized_actions,
# )
# #
# actions = actions.round().astype(int)
# #
# print(actions)
