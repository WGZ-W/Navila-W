# 检查模型目录中的配置文件
import os

model_name_or_path = "/home/weiguanzhao/navila-siglip-llama3-8b-v1.5-pretrain/llm"
config_path = os.path.join(model_name_or_path, "config.json")
if os.path.exists(config_path):
    import json
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    print("Model config:", config_data)

# 检查是否有特殊初始化设置
print("Has init_empty_weights in config:", getattr(config_data, '_init_empty_weights', False))