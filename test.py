import json
from typing import Union, List, Dict, Any
from PIL import Image
import os

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from llava.model import *

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.model.action_tokenizer import ActionTokenizer

class MultiModalProcessor:
    """
    多模态处理器，包含图像处理器和文本tokenizer
    """

    def __init__(self,
                 model,
                 max_length: int = 4096,
                 **kwargs):
        """
        初始化处理器

        Args:
            image_processor_name: 图像处理器名称或路径
            tokenizer_name: tokenizer名称或路径
            max_length: 文本最大长度
            **kwargs: 其他参数
        """
        self.image_processor = model.get_vision_tower().image_processor
        self.tokenizer = model.tokenizer
        self.max_length = max_length

        # 如果tokenizer没有pad token，设置一个
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'

    def __call__(self,
                 images: Union[Image.Image, List[Image.Image], torch.Tensor],
                 texts: Union[str, List[str]] = None,
                 padding: bool = True,
                 truncation: bool = True,
                 return_tensors: str = "pt"):
        """
        处理多模态输入

        Args:
            images: 单张图像或图像列表
            texts: 单个文本或文本列表
            padding: 是否填充
            truncation: 是否截断
            return_tensors: 返回的张量类型

        Returns:
            包含处理结果的字典
        """
        # 处理图像
        image_inputs = self.image_processor(
            images,
            return_tensors=return_tensors
        )['pixel_values']

        # 处理文本（如果提供）

        input_ids = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )['input_ids']


        return {image_inputs, input_ids}


def load_trained_model(model_path):
    """加载训练好的 LLaVA 模型"""

    # 获取模型名称
    model_name = get_model_name_from_path(model_path)

    # 加载 tokenizer、模型和图像处理器
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device='cuda:0'
    )

    return tokenizer, model, image_processor, context_len


# 使用示例
model_path = "/mnt/sdc/weiguanzhao/navila-llama3-8b-8f"  # 你的模型路径



config = LlavaLlamaConfig.from_pretrained(model_path, resume=False)
if getattr(config, "resume_path", None) is not None:
    config.resume_path = model_path

model = LlavaLlamaModel(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=4096,
    ).to("cuda:0")

processor = MultiModalProcessor(model)

image_path = "/mnt/sdc/weiguanzhao/navila/test.jpg"
image = Image.open(image_path).convert('RGB')
question = "描述这张图片中的内容。"

norm_stats = None
action_tokenizer = ActionTokenizer(model.tokenizer)

dataset_statistics_path = "/mnt/sdc/weiguanzhao/dataset_statistics.json"
if os.path.isfile(dataset_statistics_path):
    with open(dataset_statistics_path, "r") as f:
        norm_stats = json.load(f)
    model.norm_stats = norm_stats

image_tensor, input_ids = processor(image, question, return_tensors="pt")
inputs = processor(image, question, return_tensors="pt").to("cuda:0")
image_tensor, input_ids = image_tensor.to("cuda:0"), input_ids.to("cuda:0")

outputs = model.predict_action(input_ids, image_tensor, norm_stats, action_tokenizer, unnorm_key="vlnv1")

print(f"回答: {outputs}")


# tokenizer, model, image_processor, context_len = load_trained_model(model_path)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token


class MultiModalProcessor:
    """
    多模态处理器，包含图像处理器和文本tokenizer
    """

    def __init__(self,
                 model,
                 max_length: int = 4096,
                 **kwargs):
        """
        初始化处理器

        Args:
            image_processor_name: 图像处理器名称或路径
            tokenizer_name: tokenizer名称或路径
            max_length: 文本最大长度
            **kwargs: 其他参数
        """
        self.image_processor = model.image_processor
        self.tokenizer = model.tokenizer
        self.max_length = max_length

        # 如果tokenizer没有pad token，设置一个
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'

    def __call__(self,
                 images: Union[Image.Image, List[Image.Image], torch.Tensor],
                 texts: Union[str, List[str]] = None,
                 padding: bool = True,
                 truncation: bool = True,
                 return_tensors: str = "pt") -> Dict[str, Any]:
        """
        处理多模态输入

        Args:
            images: 单张图像或图像列表
            texts: 单个文本或文本列表
            padding: 是否填充
            truncation: 是否截断
            return_tensors: 返回的张量类型

        Returns:
            包含处理结果的字典
        """
        # 处理图像
        image_inputs = self.image_processor(
            images,
            return_tensors=return_tensors
        )

        # 处理文本（如果提供）
        input_ids = {}
        if texts is not None:
            input_ids = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors
            )


        return {**image_inputs, **input_ids}




def inference_with_replaced_downsample(image_path, question, tokenizer, model, image_processor):
    """使用替换后的 DownSampleBlock 进行推理"""

    from PIL import Image
    # 确保模型在正确的设备上并使用一致的数据类型
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    model = model.to(torch.bfloat16)
    main_device = model.device
    print(f"Model device: {main_device}")

    # 显式移动各个组件
    if hasattr(model, 'vision_tower'):
        model.vision_tower.to(main_device)

    if hasattr(model, 'mm_projector'):
        model.mm_projector.to(main_device)

    if hasattr(model, 'llm'):
        model.llm.to(main_device)

    # 处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_processor(image, return_tensors='pt')['pixel_values']
    image_tensor = image_tensor.to(device=main_device, dtype=model.dtype)
    print(f"Image tensor device: {image_tensor.device}")

    # 检查处理后的张量形状
    print(f"Processed image shape: {image_tensor.shape}")  # 应该是 [1, 3, H, W]

    # # 获取图像特征
    # with torch.no_grad():
    #     vision_outputs = model.vision_tower(image_tensor)
    #
    # # 检查输出类型
    # if isinstance(vision_outputs, dict):
    #     image_features = vision_outputs['last_hidden_state']
    # elif hasattr(vision_outputs, 'last_hidden_state'):
    #     image_features = vision_outputs.last_hidden_state
    # else:
    #     image_features = vision_outputs
    #
    # print(f"Extracted features shape: {image_features.shape}")
    #
    # # 确保数据类型匹配
    # if image_features.dtype != model.dtype:
    #     image_features = image_features.to(model.dtype)
    #
    # # 确保投影器使用相同的数据类型
    # mm_projector = model.mm_projector.to(model.dtype)
    #
    # # 投影特征
    # projected_features = mm_projector(image_features)

    action_tokenizer = ActionTokenizer(tokenizer)


    # 正常推理
    conv_mode = "v1"
    conv = conv_templates[conv_mode].copy()

    if model.config.mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        question = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    print(f"Input ids device: {input_ids.device}")

    # 确保模型所有部分都在同一设备上
    model = model.to(main_device)

    norm_stats = None

    dataset_statistics_path = "/mnt/sdc/weiguanzhao/dataset_statistics.json"
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        model.norm_stats = norm_stats

    print(f"type of norm {type(norm_stats)}")
    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         images=image_tensor,
    #         # image_sizes=[image.size],
    #         do_sample=True,
    #         temperature=0.2,
    #         max_new_tokens=512,
    #         use_cache=True
    #     )
    #
    # outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # return outputs
    outputs = model.predict_action(input_ids, image_tensor, norm_stats, action_tokenizer, unnorm_key="vlnv1")

    return outputs




# 准备测试
image_path = "/mnt/sdc/weiguanzhao/navila/test.jpg"
question = "描述这张图片中的内容。"



# 推理
result = inference_with_replaced_downsample(image_path, question, model.tokenizer, model, model.get_vision_tower().image_processor)
print(f"问题: {question}")
print(f"回答: {result}")
