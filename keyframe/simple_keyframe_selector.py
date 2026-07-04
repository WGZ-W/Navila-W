import torch
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
import logging
from keyframe.clip_wrapper import CLIPModelWrapper  # 假设 clip_wrapper 在同一目录下
from keyframe.clip_wrapper import SigLIPModelWrapper

logger = logging.getLogger(__name__)


class SimpleKeyframeSelector:
    """
    基于CLIP模型的简单关键帧选择器。
    直接计算整个指令文本与每张图像的相似度，选取相似度最高的若干帧作为关键帧。
    """

    def __init__(self, clip_model_name: str = "ViT-B/32", device: Optional[str] = None, threshold: float = 0.0):
        """
        初始化选择器。

        Args:
            clip_model_name: CLIP模型名称（如 "ViT-B/32"）
            device: 计算设备（"cuda" 或 "cpu"），若为None则自动选择
            threshold: 相似度阈值，低于此值的帧不会被选为关键帧
        """
        # self.clip = CLIPModelWrapper(model_name=clip_model_name, device=device)
        self.clip = SigLIPModelWrapper(
            model_path="/mnt/sdc/weiguanzhao/navila-labs/siglip-base-patch16-224",   # 本地模型文件夹路径,
            device=device
        )
        self.threshold = threshold
        self.clip.warmup()  # 预热模型，提升首次推理速度

        # 可选：记录最大文本长度（仅用于信息，非必需）
        # self.max_text_length = getattr(self.clip, 'max_text_length', 64)

    def process_images(self, images: List[Image.Image], instruction_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        处理图像列表，根据与指令的相似度选出关键帧。

        Args:
            images: PIL图像列表
            instruction_text: 英文指令文本
            top_k: 最多返回的关键帧数量

        Returns:
            包含以下字段的字典：
                - keyframe_indices: 选中的关键帧索引列表（按时间顺序）
                - keyframe_images: 对应的PIL图像列表
                - similarities: 对应关键帧的相似度分数列表
                - all_similarities: 所有图像的相似度分数列表
                - total_images: 图像总数
        """
        if not images:
            return {
                "keyframe_indices": [],
                "keyframe_images": [],
                "similarities": [],
                "all_similarities": [],
                "total_images": 0,
            }

        # 1. 编码指令文本（归一化）
        text_feature = self.clip.encode_text([instruction_text], normalize=True)  # (1, dim)

        # 2. 编码所有图像（批量）
        #    encode_image 接受 PIL Image 列表，返回 (N, dim) 张量
        image_features = self.clip.encode_image(images, normalize=True)  # (N, dim)

        # 3. 计算相似度（点积，因为特征已归一化）
        similarities = (text_feature @ image_features.T).squeeze(0).cpu().numpy()  # (N,)

        # 4. 根据阈值筛选
        valid_indices = [i for i, sim in enumerate(similarities) if sim >= self.threshold]
        valid_similarities = similarities[valid_indices]

        if len(valid_indices) == 0:
            # 没有帧达到阈值，则返回得分最高的 top_k 帧
            logger.warning("没有帧达到阈值，返回得分最高的帧")
            top_indices = np.argsort(similarities)[::-1][:top_k]
            selected_indices = sorted(top_indices.tolist())  # 保持时间顺序
            selected_similarities = [similarities[i] for i in selected_indices]
        else:
            # 对符合条件的帧按相似度降序排序，取前 top_k
            sorted_pairs = sorted(zip(valid_indices, valid_similarities), key=lambda x: x[1], reverse=True)
            top_pairs = sorted_pairs[:top_k]
            selected_indices = sorted([idx for idx, _ in top_pairs])  # 保持时间顺序
            selected_similarities = [similarities[i] for i in selected_indices]

        # 5. 提取关键帧图像
        keyframe_images = [images[i] for i in selected_indices]

        return {
            "keyframe_indices": selected_indices,
            "keyframe_images": keyframe_images,
            "similarities": selected_similarities,
            "all_similarities": similarities.tolist(),
            "total_images": len(images),
        }


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 创建测试图像（随机生成，仅用于演示）
    test_images = []
    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_images.append(img)

    instruction = "A dog is running in the park"

    # 初始化选择器
    selector = SimpleKeyframeSelector(
        clip_model_name="ViT-B/32",
        device="cpu",       # 若无GPU可设为 "cpu"
        threshold=0.2       # 相似度阈值
    )

    # 处理图像
    result = selector.process_images(test_images, instruction, top_k=3)

    print(f"选中关键帧索引: {result['keyframe_indices']}")
    print(f"对应相似度: {result['similarities']}")
    print(f"所有图像相似度: {result['all_similarities']}")