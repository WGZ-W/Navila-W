import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import logging
from PIL import Image
from dataclasses import dataclass, asdict
import time

from contextlib import contextmanager

from transformers import AutoModel, AutoProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ClipFeatureResult:
    """CLIP特征提取结果"""
    text_features: Optional[torch.Tensor] = None
    image_features: Optional[torch.Tensor] = None
    logits_per_text: Optional[torch.Tensor] = None
    logits_per_image: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for field, value in asdict(self).items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    result[field] = value.cpu().numpy()
                else:
                    result[field] = value
        return result


class CLIPModelWrapper:
    """优化的CLIP模型封装类"""

    MODEL_CONFIGS = {
        "RN50": (224, 224),
        "RN101": (224, 224),
        "RN50x4": (288, 288),
        "RN50x16": (384, 384),
        "RN50x64": (448, 448),
        "ViT-B/32": (224, 224),
        "ViT-B/16": (224, 224),
        "ViT-L/14": (224, 224),
        "ViT-L/14@336px": (336, 336),
    }

    def __init__(
            self,
            model_name: str = "ViT-B/32",
            device: Optional[str] = None,
            model_path: Optional[str] = None,
            download_root: Optional[str] = None,
            jit: bool = False,
            cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.jit = jit
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/clip")

        # 设置设备
        self.device = self._setup_device(device)

        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)

        # 加载模型
        self.model, self.preprocess = self._load_model(model_path, download_root)
        self.model = self.model.to(self.device).eval()

        # 获取输入尺寸
        self.input_size = self.MODEL_CONFIGS.get(model_name, (224, 224))
        logger.info(f"CLIP模型初始化完成: {model_name}, 输入尺寸: {self.input_size}")

    def _setup_device(self, device_str: Optional[str]) -> torch.device:
        """设置计算设备"""
        if device_str:
            return torch.device(device_str)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_path: Optional[str], download_root: Optional[str]) -> Any:
        """加载CLIP模型"""
        try:
            import clip

            if download_root:
                os.environ['CLIP_MODEL_PATH'] = download_root

            if model_path and os.path.exists(model_path):
                logger.info(f"从本地加载模型: {model_path}")
                return clip.load(model_path, device=self.device, jit=self.jit)
            else:
                logger.info(f"下载CLIP模型: {self.model_name}")
                return clip.load(
                    self.model_name,
                    device=self.device,
                    jit=self.jit,
                    download_root=download_root
                )

        except ImportError:
            raise ImportError(
                "请安装CLIP库: pip install git+https://github.com/openai/CLIP.git"
            )

    def _process_single_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """处理单张图像"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            image = Image.fromarray(image.astype(np.uint8))

        if isinstance(image, Image.Image):
            return self.preprocess(image)
        elif isinstance(image, torch.Tensor):
            return image
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]], normalize: bool = True) -> torch.Tensor:
        """编码文本为特征向量"""
        import clip

        if isinstance(texts, str):
            texts = [texts]

        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
        text_features = self.model.encode_text(text_tokens)

        if normalize:
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    @torch.no_grad()
    def encode_image(self, images: Union[Image.Image, np.ndarray, List, torch.Tensor],
                     normalize: bool = True) -> torch.Tensor:
        """编码图像为特征向量"""
        if not isinstance(images, list):
            images = [images]

        # 处理每张图像
        processed_images = []
        for img in images:
            try:
                processed = self._process_single_image(img).unsqueeze(0)
                processed_images.append(processed)
            except Exception as e:
                logger.warning(f"图像处理失败: {e}")
                # 使用黑色图像作为占位符
                placeholder = torch.zeros(1, 3, *self.input_size)
                processed_images.append(placeholder)

        image_tensor = torch.cat(processed_images, dim=0).to(self.device)
        image_features = self.model.encode_image(image_tensor)

        if normalize:
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    @torch.no_grad()
    def encode_batch(
            self,
            images: Optional[List] = None,
            texts: Optional[List[str]] = None,
            normalize: bool = True
    ) -> ClipFeatureResult:
        """批量编码图像和文本"""
        result = ClipFeatureResult()

        if texts:
            result.text_features = self.encode_text(texts, normalize)

        if images:
            result.image_features = self.encode_image(images, normalize)

        # 计算相似度
        if result.text_features is not None and result.image_features is not None:
            logit_scale = self.model.logit_scale.exp()
            similarity = (result.text_features @ result.image_features.T) * logit_scale

            result.logits_per_text = similarity
            result.logits_per_image = similarity.T
            result.probs = F.softmax(similarity, dim=-1)

        return result

    @torch.no_grad()
    def predict(
            self,
            image: Union[np.ndarray, Image.Image],
            text_options: List[str],
            top_k: int = 5
    ) -> Dict[str, Any]:
        """预测图像最匹配的文本"""
        if not text_options:
            raise ValueError("text_options不能为空")

        # 编码
        result = self.encode_batch(images=[image], texts=text_options)

        if result.probs is None or result.logits_per_image is None:
            raise ValueError("编码失败")

        # 获取概率和logits
        # result.probs形状: [n_text, 1]
        # result.logits_per_image形状: [1, n_text]
        probs = result.probs.squeeze(1)  # 形状: [n_text]
        logits = result.logits_per_image.squeeze(0)  # 形状: [n_text]

        # 确保k不超过文本数量
        k = min(top_k, len(text_options))
        top_probs, top_indices = probs.topk(k)

        predictions = [
            {
                "text": text_options[idx],
                "probability": float(prob),
                "logit": float(logits[idx])
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

        return {
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
            "logits": logits.cpu().numpy(),
            "probs": probs.cpu().numpy()
        }

    def extract_features_batch(
            self,
            items: List,
            is_image: bool = True,
            batch_size: int = 32,
            show_progress: bool = True
    ) -> torch.Tensor:
        """批量提取特征"""
        from tqdm import tqdm

        all_features = []

        pbar = tqdm(total=len(items), desc="提取特征", disable=not show_progress)

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            try:
                if is_image:
                    features = self.encode_image(batch)
                else:
                    features = self.encode_text(batch)

                all_features.append(features.cpu())
            except Exception as e:
                logger.error(f"批次 {i // batch_size} 失败: {e}")
                # 添加零特征作为占位符
                feat_dim = 512  # CLIP特征维度
                placeholder = torch.zeros(len(batch), feat_dim)
                all_features.append(placeholder)

            pbar.update(len(batch))

        pbar.close()

        return torch.cat(all_features, dim=0) if all_features else torch.tensor([])

    def save_features(self, features: torch.Tensor, file_path: str, metadata: Optional[Dict] = None):
        """保存特征到文件"""
        data = {
            "features": features.cpu().numpy(),
            "metadata": metadata or {},
            "timestamp": time.time()
        }

        if file_path.endswith(".npy"):
            np.save(file_path, data, allow_pickle=True)
        elif file_path.endswith(".npz"):
            np.savez(file_path, **data)
        elif file_path.endswith(".pt"):
            torch.save(data, file_path)
        else:
            file_path = f"{file_path}.npy"
            np.save(file_path, data, allow_pickle=True)

        logger.info(f"特征已保存到: {file_path}")

    @staticmethod
    def load_features(file_path: str) -> Dict[str, Any]:
        """从文件加载特征"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if file_path.endswith(".npy"):
            data = np.load(file_path, allow_pickle=True).item()
        elif file_path.endswith(".npz"):
            data = dict(np.load(file_path, allow_pickle=True))
        elif file_path.endswith(".pt"):
            data = torch.load(file_path, map_location="cpu")
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

        # 转换为张量
        if "features" in data and isinstance(data["features"], np.ndarray):
            data["features"] = torch.from_numpy(data["features"])

        return data

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "input_size": self.input_size,
            "jit": self.jit,
            "cache_dir": self.cache_dir
        }

        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        info["total_params"] = total_params

        return info

    def warmup(self, n_iterations: int = 3):
        """预热模型"""
        logger.info(f"预热模型 ({n_iterations} 次迭代)")

        dummy_text = ["warmup text"]
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        for i in range(n_iterations):
            _ = self.encode_text(dummy_text)
            _ = self.encode_image(dummy_image)

        logger.info("模型预热完成")



class SigLIPModelWrapper:
    """SigLIP模型封装类，替换原有的CLIPModelWrapper"""
    MODEL_CONFIGS = {
        "google/siglip-base-patch16-224": (224, 224),
        "google/siglip-large-patch16-256": (256, 256),
        "google/siglip-so400m-patch14-384": (384, 384),
        # ... 可添加其他SigLIP模型
    }

    def __init__(
            self,
            model_name: str = "google/siglip-base-patch16-224",
            device: Optional[str] = None,
            # model_path: Optional[str] = None,  # 此参数在使用AutoModel时可能不再需要
            model_path: Optional[str] = None,  # 新增：本地模型路径
            jit: bool = False,  # SigLIP通常不使用jit
            cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        # 设置设备
        self.device = self._setup_device(device)
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)

        # 关键修改：优先使用本地路径
        model_id = model_path if model_path is not None else model_name

        # 加载模型和处理器
        # 注意：model_path参数在此被忽略，因为AutoModel总是从Hugging Face Hub或其缓存加载
        # 加载模型和处理器
        self.model = AutoModel.from_pretrained(
            model_id,
            cache_dir=self.cache_dir
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=self.cache_dir
        )

        # self.model = AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device).eval()
        # self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)

        self.max_text_length = self.model.config.text_config.max_position_embeddings

        if hasattr(self.processor, 'size'):
            self.input_size = (self.processor.size['height'], self.processor.size['width'])
        else:
            # 方法2：从模型配置中获取
            self.input_size = self.MODEL_CONFIGS.get(model_name, (224, 224))

        logger.info(f"SigLIP模型初始化完成: {model_id}, 输入尺寸: {self.input_size}")

        # 获取输入尺寸
        # self.input_size = self.MODEL_CONFIGS.get(model_name, (224, 224))
        # logger.info(f"SigLIP模型初始化完成: {model_name}, 输入尺寸: {self.input_size}")

    # _setup_device, get_model_info, warmup等方法与原有CLIPModelWrapper相同，无需修改
    # ...

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]], normalize: bool = True) -> torch.Tensor:
        """编码文本为特征向量"""
        if isinstance(texts, str):
            texts = [texts]

        # 使用processor处理文本
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        ).to(self.device)
        text_features = self.model.get_text_features(**inputs)

        if normalize:
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    @torch.no_grad()
    def encode_image(self, images: Union[Image.Image, np.ndarray, List, torch.Tensor],
                     normalize: bool = True) -> torch.Tensor:
        """编码图像为特征向量"""
        if not isinstance(images, list):
            images = [images]

        # 确保所有图像都是PIL Image格式
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[2] == 4:
                    img = img[:, :, :3]
                img = Image.fromarray(img.astype(np.uint8))
            pil_images.append(img)

        # 批量处理图像
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)

        if normalize:
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    @torch.no_grad()
    def encode_batch(self, images: Optional[List] = None, texts: Optional[List[str]] = None,
                     normalize: bool = True) -> 'ClipFeatureResult':
        """批量编码图像和文本"""
        result = ClipFeatureResult()

        if texts:
            result.text_features = self.encode_text(texts, normalize)
        if images:
            result.image_features = self.encode_image(images, normalize)

        # 计算相似度: SigLIP的输出特征已是L2归一化的，直接使用余弦相似度计算匹配概率
        if result.text_features is not None and result.image_features is not None:
            # 方法1：计算logits（相似度得分），用于需要logit值的场景
            # logits_per_text形状: [n_text, n_image]
            logits_per_text = (result.text_features @ result.image_features.T)
            result.logits_per_text = logits_per_text
            result.logits_per_image = logits_per_text.T

            # 方法2：计算概率（通过sigmoid）
            # probs_per_image形状: [n_image, n_text]
            probs = torch.sigmoid(result.logits_per_image)
            result.probs = probs

        return result

    # 在 SigLIPModelWrapper 类中添加以下方法

    def _setup_device(self, device_str: Optional[str]) -> torch.device:
        """设置计算设备"""
        if device_str:
            return torch.device(device_str)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "input_size": self.input_size,
            "jit": getattr(self, 'jit', False),
            "cache_dir": self.cache_dir
        }
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        info["total_params"] = total_params
        # 获取特征维度（SigLIP）
        with torch.no_grad():
            dummy_input = self.processor(text=["test"], return_tensors="pt").to(self.device)
            dummy_feat = self.model.get_text_features(**dummy_input)
            info["feature_dim"] = dummy_feat.shape[-1]
        return info

    def warmup(self, n_iterations: int = 3):
        """预热模型"""
        logger.info(f"预热模型 ({n_iterations} 次迭代)")
        dummy_text = ["warmup text"]
        dummy_image = Image.new('RGB', self.input_size, color='black')
        for _ in range(n_iterations):
            _ = self.encode_text(dummy_text)
            _ = self.encode_image(dummy_image)
        logger.info("模型预热完成")

    @torch.no_grad()
    def predict(self, image: Union[np.ndarray, Image.Image], text_options: List[str], top_k: int = 5) -> Dict[str, Any]:
        """预测图像最匹配的文本（与原有CLIPModelWrapper.predict逻辑相同）"""
        if not text_options:
            raise ValueError("text_options不能为空")
        result = self.encode_batch(images=[image], texts=text_options)
        if result.probs is None or result.logits_per_image is None:
            raise ValueError("编码失败")
        # result.probs形状: [1, n_text]
        probs = result.probs.squeeze(0)  # [n_text]
        logits = result.logits_per_image.squeeze(0)  # [n_text]
        k = min(top_k, len(text_options))
        top_probs, top_indices = probs.topk(k)
        predictions = [
            {
                "text": text_options[idx],
                "probability": float(prob),
                "logit": float(logits[idx])
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        return {
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
            "logits": logits.cpu().numpy(),
            "probs": probs.cpu().numpy()
        }

    def extract_features_batch(self, items: List, is_image: bool = True, batch_size: int = 32,
                               show_progress: bool = True) -> torch.Tensor:
        """批量提取特征（注意：特征维度需动态获取）"""
        from tqdm import tqdm
        all_features = []
        # 动态获取特征维度
        if is_image:
            dummy_item = items[0] if items else None
            if dummy_item is None:
                return torch.tensor([])
            with torch.no_grad():
                dummy_feat = self.encode_image([dummy_item])  # [1, dim]
                feat_dim = dummy_feat.shape[-1]
        else:
            dummy_item = items[0] if items else "test"
            with torch.no_grad():
                dummy_feat = self.encode_text([dummy_item])
                feat_dim = dummy_feat.shape[-1]
        pbar = tqdm(total=len(items), desc="提取特征", disable=not show_progress)
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            try:
                if is_image:
                    features = self.encode_image(batch)
                else:
                    features = self.encode_text(batch)
                all_features.append(features.cpu())
            except Exception as e:
                logger.error(f"批次 {i // batch_size} 失败: {e}")
                placeholder = torch.zeros(len(batch), feat_dim)
                all_features.append(placeholder)
            pbar.update(len(batch))
        pbar.close()
        return torch.cat(all_features, dim=0) if all_features else torch.tensor([])

    def save_features(self, features: torch.Tensor, file_path: str, metadata: Optional[Dict] = None):
        """保存特征到文件（与原方法完全相同）"""
        data = {
            "features": features.cpu().numpy(),
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        if file_path.endswith(".npy"):
            np.save(file_path, data, allow_pickle=True)
        elif file_path.endswith(".npz"):
            np.savez(file_path, **data)
        elif file_path.endswith(".pt"):
            torch.save(data, file_path)
        else:
            file_path = f"{file_path}.npy"
            np.save(file_path, data, allow_pickle=True)
        logger.info(f"特征已保存到: {file_path}")

    @staticmethod
    def load_features(file_path: str) -> Dict[str, Any]:
        """从文件加载特征（与原方法完全相同）"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if file_path.endswith(".npy"):
            data = np.load(file_path, allow_pickle=True).item()
        elif file_path.endswith(".npz"):
            data = dict(np.load(file_path, allow_pickle=True))
        elif file_path.endswith(".pt"):
            data = torch.load(file_path, map_location="cpu")
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        if "features" in data and isinstance(data["features"], np.ndarray):
            data["features"] = torch.from_numpy(data["features"])
        return data




class ZeroShotClassifier:
    """零样本分类器"""

    def __init__(self, clip_model: CLIPModelWrapper, template: str = "a photo of a {}"):
        self.clip = clip_model
        self.template = template
        self.classes: Optional[List[str]] = None
        self.features: Optional[torch.Tensor] = None

    def set_classes(self, classes: List[str]):
        """设置分类类别"""
        self.classes = classes
        prompts = [self.template.format(c) for c in classes]
        self.features = self.clip.encode_text(prompts)
        logger.info(f"设置了 {len(classes)} 个类别")

    @torch.no_grad()
    def classify(self, image: Union[np.ndarray, Image.Image], top_k: int = 5) -> Dict[str, Any]:
        """分类单张图像"""
        if self.features is None or self.classes is None:
            raise ValueError("请先调用 set_classes()")

        # 提取图像特征
        image_feat = self.clip.encode_image(image)

        # 计算相似度
        logit_scale = self.clip.model.logit_scale.exp()
        similarity = (self.features @ image_feat.T) * logit_scale

        # 计算概率
        probs = F.softmax(similarity, dim=0).squeeze()  # 形状: [n_classes]
        logits = similarity.squeeze()  # 形状: [n_classes]

        # 获取top-k
        k = min(top_k, len(self.classes))
        top_probs, top_indices = probs.topk(k)

        predictions = [
            {
                "class": self.classes[idx],
                "probability": float(prob),
                "logit": float(logits[idx])
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

        return {
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
            "probs": probs.cpu().numpy()
        }


# 使用示例
if __name__ == "__main__":
    print("=== CLIP模型测试 ===")

    try:
        # 创建模型
        clip_model = CLIPModelWrapper(model_name="ViT-B/32")

        # 模型信息
        print("模型信息:")
        for k, v in clip_model.get_model_info().items():
            print(f"  {k}: {v}")

        # 测试文本编码
        texts = ["a cat", "a dog", "a car"]
        text_feats = clip_model.encode_text(texts)
        print(f"\n文本特征形状: {text_feats.shape}")

        # 测试图像编码
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_feat = clip_model.encode_image(test_img)
        print(f"图像特征形状: {img_feat.shape}")

        # 测试预测
        result = clip_model.predict(test_img, texts)
        print(f"\n预测结果:")
        for pred in result["predictions"]:
            print(f"  {pred['text']}: {pred['probability']:.4f}")

        # 测试零样本分类
        classifier = ZeroShotClassifier(clip_model)
        classifier.set_classes(["cat", "dog", "bird", "car", "tree", "house"])
        cls_result = classifier.classify(test_img)
        print(f"\n分类结果:")
        for pred in cls_result["predictions"]:
            print(f"  {pred['class']}: {pred['probability']:.4f}")

        print("\n✅ 测试通过!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()



# 在你的关键帧选择器中集成CLIP
class EnhancedKeyframeSelector:
    def __init__(self, clip_model_path=None):
        # 初始化CLIP模型
        self.clip_model = CLIPModelWrapper(
            model_name="ViT-B/32",
            model_path=clip_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 预热模型
        self.clip_model.warmup()

    def extract_noun_features(self, nouns_list):
        """为名词列表提取CLIP特征"""
        texts = [noun["text"] for noun in nouns_list]
        text_features = self.clip_model.encode_text(texts)

        # 将特征添加到名词字典中
        for i, noun in enumerate(nouns_list):
            noun["clip_feature"] = text_features[i]

        return nouns_list

    def compute_frame_similarity(self, frame_features, noun_features):
        """计算帧特征和名词特征的相似度"""
        # 假设frame_features形状为 [n_frames, feature_dim]
        # noun_features形状为 [n_nouns, feature_dim]
        similarity = self.clip_model.compute_similarity(noun_features, frame_features)
        return similarity

    def detect_objects_in_frame(self, frame, detector, clip_model):
        """检测帧中的物体并提取CLIP特征"""
        # 使用YOLO检测物体
        detections = detector.detect(frame, extract_crops=True)

        object_features = []
        object_info = []

        for det in detections:
            if det.cropped_image is not None:
                # 提取物体图像的CLIP特征
                obj_feature = clip_model.encode_image(det.cropped_image)
                object_features.append(obj_feature)

                object_info.append({
                    "bbox": det.bbox,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "feature": obj_feature
                })

        if object_features:
            object_features = torch.stack(object_features, dim=0)

        return object_features, object_info