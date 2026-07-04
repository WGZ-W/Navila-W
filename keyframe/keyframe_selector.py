# image_keyframe_selector.py
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import logging
from PIL import Image

# 导入各个模块
from keyframe.clip_wrapper import CLIPModelWrapper
from keyframe.noun_extractor import EnglishNounExtractor
from keyframe.object_detector import YOLODetector
from keyframe.clip_wrapper import SigLIPModelWrapper

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """帧数据容器"""
    frame_idx: int
    image: Image.Image
    image_feature: torch.Tensor
    detected_objects: List[Dict]


@dataclass
class NounInfo:
    """名词信息容器"""
    text: str
    type: str  # "scene" 或 "object"
    feature: torch.Tensor
    confidence: float = 1.0
    best_frame_idx: Optional[int] = None
    best_similarity: float = 0.0


class KeyframeSelector:
    """
    图像关键帧选择器 - 直接处理图像列表
    """

    def __init__(
            self,
            clip_model_name: str = "ViT-B/32",
            clip_model_path: Optional[str] = None,
            detector_model_type: str = "yolov8",
            detector_model_path: Optional[str] = None,
            device: Optional[str] = None,
            threshold: float = 0.3,
            batch_size: int = 32
    ):
        """
        初始化图像关键帧选择器

        参数:
            clip_model_name: CLIP模型名称
            clip_model_path: CLIP模型路径
            detector_model_type: 检测器类型 ('yolov8' 或 'yolov5')
            detector_model_path: 检测器模型路径
            device: 计算设备 ('cuda' 或 'cpu')
            threshold: 相似度阈值
            batch_size: 批量处理大小
        """
        self.threshold = threshold
        self.batch_size = batch_size

        # 自动选择设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化各模块
        # logger.info("初始化CLIP模型...")
        # self.clip_model = CLIPModelWrapper(
        #     model_name=clip_model_name,
        #     model_path=clip_model_path,
        #     device=device
        # )

        self.clip_model = SigLIPModelWrapper(
            model_name="google/siglip-base-patch16-224", device=device)

        # logger.info("预热CLIP模型...")
        self.clip_model.warmup()

        # logger.info("初始化物体检测器...")
        self.detector = YOLODetector(
            model_type=detector_model_type,
            model_path=detector_model_path,
            device=device,
            conf_threshold=0.25  # 较低的阈值以检测更多物体
        )

        # logger.info("初始化名词提取器...")
        self.noun_extractor = EnglishNounExtractor()

        # logger.info(f"图像关键帧选择器初始化完成 (阈值: {threshold}, 设备: {device})")

    def extract_nouns_from_text(self, instruction_text: str) -> List[NounInfo]:
        """
        步骤1: 名词提取与处理
        从指令文本中提取名词短语，并区分场景和物体
        """
        # logger.info(f"从指令文本中提取名词: {instruction_text[:50]}...")

        # 使用名词提取器
        noun_dicts = self.noun_extractor.extract_nouns(instruction_text)

        if not noun_dicts:
            logger.warning("未提取到名词，尝试备用策略...")
            # 备用策略：简单分词
            import re
            words = re.findall(r'\b[a-zA-Z]{3,}\b', instruction_text)
            noun_dicts = [
                {"text": word, "type": "object", "confidence": 0.5}
                for word in words[:10]  # 最多取10个
            ]

        # 为名词编码CLIP特征
        noun_texts = [n["text"] for n in noun_dicts]
        # logger.info(f"提取到 {len(noun_texts)} 个名词: {noun_texts}")

        # 批量编码文本特征
        text_features = self.clip_model.encode_text(noun_texts, normalize=True)

        # 创建NounInfo对象
        nouns = []
        for i, noun_dict in enumerate(noun_dicts):
            noun = NounInfo(
                text=noun_dict["text"],
                type=noun_dict["type"],
                feature=text_features[i],
                confidence=noun_dict["confidence"]
            )
            nouns.append(noun)
            # logger.debug(f"名词: {noun.text} ({noun.type}), 置信度: {noun.confidence:.2f}")

        return nouns

    def process_single_image(self, image: Image.Image, frame_idx: int) -> FrameData:
        """
        处理单张 PIL 图像：
        1. 提取整图CLIP特征
        2. 使用检测器检测物体
        3. 为检测到的物体提取CLIP特征
        """

        # 将PIL Image转换为numpy数组用于物体检测
        image_np = np.array(image)

        # 确保图像是RGB格式
        if len(image_np.shape) == 2:  # 灰度图
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 4:  # RGBA
            image_np = image_np[:, :, :3]

        # 提取整图特征
        image_feature = self.clip_model.encode_image(image_np, normalize=True)

        # 物体检测
        try:
            detections = self.detector.detect(image_np)

            # 为每个检测到的物体提取特征
            detected_objects = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']

                # 提取物体区域
                obj_image_np = image_np[y1:y2, x1:x2]

                if obj_image_np.size > 0 and obj_image_np.shape[0] > 5 and obj_image_np.shape[1] > 5:
                    # 提取物体特征
                    obj_feature = self.clip_model.encode_image(obj_image_np, normalize=True)

                    detected_objects.append({
                        "bbox": det['bbox'],
                        "confidence": det['confidence'],
                        "class_id": det['class_id'],
                        "class_name": det['class_name'],
                        "feature": obj_feature.squeeze(0)  # 移除批次维度
                    })
        except Exception as e:
            logger.warning(f"图像 {frame_idx} 物体检测失败: {e}")
            detected_objects = []

        return FrameData(
            frame_idx=frame_idx,
            image=image,
            image_feature=image_feature.squeeze(0),  # 移除批次维度
            detected_objects=detected_objects
        )

    def process_images_batch(self, images: List[Image.Image]) -> List[FrameData]:
        """
        批量处理图像列表

        参数:
            images: 图像列表，每个元素为numpy数组 (H, W, 3)

        返回:
            处理后的帧数据列表
        """
        # logger.info(f"处理 {len(images)} 张图像...")

        frames_data = []

        # 分批处理
        from tqdm import tqdm
        # pbar = tqdm(total=len(images), desc="处理图像")

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]

            # 批量转换为numpy数组
            batch_images_np = [np.array(img) for img in batch_images]

            # 批量编码整图特征
            try:
                batch_features = self.clip_model.encode_image(batch_images_np, normalize=True)
            except Exception as e:
                logger.error(f"批量编码特征失败: {e}")
                batch_features = None

            for j, (image, image_np) in enumerate(zip(batch_images, batch_images_np)):
                frame_idx = i + j
                try:
                    # 如果批量编码成功，使用批量特征
                    image_feature = batch_features[j] if batch_features is not None else None

                    if image_feature is None:
                        # 回退到单张编码
                        image_feature = self.clip_model.encode_image(image_np, normalize=True)

                    # 物体检测
                    detections = self.detector.detect(image_np)

                    # 为每个检测到的物体提取特征
                    detected_objects = []
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']

                        # 提取物体区域
                        obj_image_np = image_np[y1:y2, x1:x2]

                        if obj_image_np.size > 0 and obj_image_np.shape[0] > 5 and obj_image_np.shape[1] > 5:
                            # 直接使用numpy数组提取物体特征
                            obj_feature = self.clip_model.encode_image(obj_image_np, normalize=True)

                            detected_objects.append({
                                "bbox": det['bbox'],
                                "confidence": det['confidence'],
                                "class_id": det['class_id'],
                                "class_name": det['class_name'],
                                "feature": obj_feature.squeeze(0)  # 移除批次维度
                            })

                    frame_data = FrameData(
                        frame_idx=frame_idx,
                        image=image,  # 保留原始PIL Image
                        image_feature=image_feature.squeeze(0),
                        detected_objects=detected_objects
                    )
                    frames_data.append(frame_data)

                except Exception as e:
                    logger.error(f"处理图像 {frame_idx} 失败: {e}")
                    # 添加空的帧数据作为占位符
                    frames_data.append(FrameData(
                        frame_idx=frame_idx,
                        image=image,
                        image_feature=torch.zeros(512),  # CLIP特征维度
                        detected_objects=[]
                    ))

                # pbar.update(1)

        # pbar.close()

        # logger.info(f"完成处理 {len(frames_data)} 张图像")
        return frames_data

    def compute_similarity(self, feature1: torch.Tensor, feature2: torch.Tensor) -> float:
        """
        计算两个特征的余弦相似度
        """
        if feature1.dim() == 1:
            feature1 = feature1.unsqueeze(0)
        if feature2.dim() == 1:
            feature2 = feature2.unsqueeze(0)

        similarity = F.cosine_similarity(feature1, feature2)
        return similarity.item()

    def compute_similarity_matrix(self, nouns: List[NounInfo], frames_data: List[FrameData]) -> np.ndarray:
        """
        步骤3: 名词-图像相似度矩阵计算

        参数:
            nouns: 名词列表
            frames_data: 帧数据列表

        返回:
            similarity_matrix: M×N相似度矩阵，M为名词数，N为图像数
        """
        M = len(nouns)
        N = len(frames_data)

        # logger.info(f"计算相似度矩阵 ({M} 名词 × {N} 图像)...")

        # 初始化相似度矩阵
        similarity_matrix = np.zeros((M, N))

        from tqdm import tqdm

        # 为每个名词计算相似度
        # for k, noun in enumerate(tqdm(nouns, desc="计算名词相似度")):
        for k, noun in enumerate(nouns):
            noun_feature = noun.feature

            for i, frame_data in enumerate(frames_data):
                if noun.type == "scene":
                    # 场景名词：与整图特征计算相似度
                    frame_feature = frame_data.image_feature
                    similarity = self.compute_similarity(noun_feature, frame_feature)
                    similarity_matrix[k, i] = similarity

                else:  # 物体名词
                    # 初始化最高分
                    max_similarity = 0

                    # 遍历图像中检测到的所有物体
                    for obj in frame_data.detected_objects:
                        obj_feature = obj["feature"]
                        similarity = self.compute_similarity(noun_feature, obj_feature)

                        if similarity > max_similarity:
                            max_similarity = similarity

                    similarity_matrix[k, i] = max_similarity

        # logger.info(f"相似度矩阵计算完成，形状: {similarity_matrix.shape}")
        return similarity_matrix

    def select_keyframes_for_nouns(self, nouns: List[NounInfo], similarity_matrix: np.ndarray,
                                   frames_data: List[FrameData]) -> List[int]:
        """
        步骤4: 为每个名词选择关键帧

        返回:
            keyframe_indices: 关键帧索引列表（按时间顺序，非重复）
        """
        # logger.info("为每个名词选择关键帧...")

        selected_frames = {}  # 名词 -> (帧索引, 得分)

        for k, noun in enumerate(nouns):
            # 获取该名词在所有图像上的得分序列
            scores = similarity_matrix[k, :]

            # 找到最大得分及其图像索引
            max_score = np.max(scores)
            max_index = np.argmax(scores)

            # 检查是否达到阈值
            if max_score >= self.threshold:
                # 更新名词的最佳帧信息
                noun.best_frame_idx = max_index
                noun.best_similarity = max_score

                # 存储帧索引和得分
                selected_frames[noun.text] = {
                    "frame_idx": max_index,
                    "score": max_score,
                    "type": noun.type
                }

                # logger.info(f"名词 '{noun.text}' ({noun.type}): 选择图像 {max_index}, 得分 {max_score:.3f}")
            # else:
                # logger.warning(
                    # f"名词 '{noun.text}' ({noun.type}): 得分 {max_score:.3f} 低于阈值 {self.threshold}, 丢弃")

        # 提取所有选中的图像索引
        frame_indices = [info["frame_idx"] for info in selected_frames.values()]

        # 去重并按照顺序排序
        unique_sorted_indices = sorted(set(frame_indices))

        # 如果没有任何图像被选中，选择得分最高的前3帧
        if not unique_sorted_indices and len(frames_data) > 0:
            # logger.warning("没有图像达到阈值，选择全局得分最高的3帧")
            # 计算每帧的平均得分
            frame_scores = similarity_matrix.mean(axis=0)
            top_indices = np.argsort(frame_scores)[-4:][::-1]
            unique_sorted_indices = sorted(top_indices.tolist())

        # logger.info(f"最终选择 {len(unique_sorted_indices)} 个关键帧: {unique_sorted_indices}")
        return unique_sorted_indices

    def get_noun_scores_summary(self, nouns: List[NounInfo], similarity_matrix: np.ndarray) -> Dict[str, Any]:
        """
        获取名词得分统计摘要
        """
        summary = {
            "nouns": [],
            "statistics": {
                "total_nouns": len(nouns),
                "nouns_above_threshold": 0,
                "average_max_score": 0,
                "median_max_score": 0
            }
        }

        max_scores = []

        for k, noun in enumerate(nouns):
            scores = similarity_matrix[k, :]
            max_score = np.max(scores) if len(scores) > 0 else 0

            noun_info = {
                "text": noun.text,
                "type": noun.type,
                "confidence": noun.confidence,
                "max_score": float(max_score),
                "above_threshold": max_score >= self.threshold
            }

            summary["nouns"].append(noun_info)
            max_scores.append(max_score)

            if max_score >= self.threshold:
                summary["statistics"]["nouns_above_threshold"] += 1

        if max_scores:
            summary["statistics"]["average_max_score"] = float(np.mean(max_scores))
            summary["statistics"]["median_max_score"] = float(np.median(max_scores))

        return summary

    def process_images(self, images: List[Image.Image], instruction_text: str) -> Dict[str, Any]:
        """
        完整的处理流程 - 直接处理图像列表

        参数:
            images: 图像列表，每个元素为numpy数组 (H, W, 3)
            instruction_text: 指令文本

        返回:
            包含所有结果的字典
        """
        import time
        start_time = time.time()

        # logger.info(f"图像数量: {len(images)}")
        # logger.info(f"指令: {instruction_text[:50]}...")
        # logger.info("=" * 60)

        # 步骤1: 名词提取与处理
        # logger.info("步骤1: 名词提取与处理...")
        nouns = self.extract_nouns_from_text(instruction_text)

        if not nouns:
            logger.error("未提取到有效名词，无法继续处理")
            return {"error": "未提取到有效名词"}

        # 步骤2: 图像处理
        # logger.info("步骤2: 图像处理...")
        frames_data = self.process_images_batch(images)

        if not frames_data:
            # logger.error("无法处理图像")
            return {"error": "无法处理图像"}

        # 步骤3: 计算相似度矩阵
        # logger.info("步骤3: 计算相似度矩阵...")
        similarity_matrix = self.compute_similarity_matrix(nouns, frames_data)

        # 步骤4: 选择关键帧
        # logger.info("步骤4: 选择关键帧...")
        keyframe_indices = self.select_keyframes_for_nouns(nouns, similarity_matrix, frames_data)

        # 计算处理时间
        processing_time = time.time() - start_time

        # 获取名词得分摘要
        noun_summary = self.get_noun_scores_summary(nouns, similarity_matrix)

        # 准备结果
        result = {
            "nouns": [
                {
                    "text": noun.text,
                    "type": noun.type,
                    "confidence": noun.confidence,
                    "best_frame_idx": noun.best_frame_idx,
                    "best_similarity": noun.best_similarity
                }
                for noun in nouns
            ],
            "keyframe_indices": keyframe_indices,
            "keyframe_images": [images[idx] for idx in keyframe_indices] if keyframe_indices else [],
            "total_images": len(frames_data),
            "processing_time": processing_time,
            "noun_summary": noun_summary,
            "similarity_matrix_shape": similarity_matrix.shape
        }

        # logger.info(f"处理完成，耗时: {processing_time:.2f}秒")
        # logger.info(f"选择的关键帧索引: {keyframe_indices}")

        return result

    def process_images_with_features(self, image_features: List[torch.Tensor],
                                     object_features_list: List[List[torch.Tensor]],
                                     instruction_text: str) -> Dict[str, Any]:
        """
        处理已经提取好特征的图像

        参数:
            image_features: 整图特征列表，每个元素形状为 [feature_dim]
            object_features_list: 物体特征列表的列表，每个元素是物体特征列表
            instruction_text: 指令文本

        返回:
            包含所有结果的字典
        """
        import time
        start_time = time.time()

        # logger.info("开始处理预提取特征的图像...")

        # 步骤1: 名词提取与处理
        nouns = self.extract_nouns_from_text(instruction_text)

        if not nouns:
            logger.error("未提取到有效名词，无法继续处理")
            return {"error": "未提取到有效名词"}

        # 创建简化的帧数据
        frames_data = []
        for i, (img_feat, obj_feats) in enumerate(zip(image_features, object_features_list)):
            detected_objects = []
            for j, obj_feat in enumerate(obj_feats):
                detected_objects.append({
                    "feature": obj_feat,
                    "class_name": f"object_{j}"
                })

            frames_data.append(FrameData(
                frame_idx=i,
                image=np.zeros((224, 224, 3)),  # 占位符
                image_feature=img_feat,
                detected_objects=detected_objects
            ))

        # 步骤3: 计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix(nouns, frames_data)

        # 步骤4: 选择关键帧
        keyframe_indices = self.select_keyframes_for_nouns(nouns, similarity_matrix, frames_data)

        # 计算处理时间
        processing_time = time.time() - start_time

        # 准备结果
        result = {
            "nouns": [
                {
                    "text": noun.text,
                    "type": noun.type,
                    "best_frame_idx": noun.best_frame_idx,
                    "best_similarity": noun.best_similarity
                }
                for noun in nouns
            ],
            "keyframe_indices": keyframe_indices,
            "total_images": len(frames_data),
            "processing_time": processing_time,
            "similarity_matrix_shape": similarity_matrix.shape
        }

        # logger.info(f"处理完成，耗时: {processing_time:.2f}秒")

        return result



# 使用示例
if __name__ == "__main__":
    print("=== 图像关键帧选择系统测试 ===")

    # 创建测试图像列表
    num_test_images = 20
    test_images = []
    for i in range(num_test_images):
        # 创建随机测试图像
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        test_images.append(img)

    # 测试指令
    instruction_text = "A dog is chasing a ball in the park"

    print("\n方法1: 使用完整版选择器")
    selector = KeyframeSelector(
        clip_model_name="ViT-B/32",
        detector_model_type="yolov8",
        threshold=0.2,  # 较低的阈值
        device="cpu"  # 使用CPU进行测试
    )

    result = selector.process_images(test_images, instruction_text)
    print(f"选择的关键帧索引: {result['keyframe_indices']}")


    # 方法3: 测试处理预提取特征
    print("\n方法3: 测试处理预提取特征")

    # 模拟预提取特征
    num_images = 10
    feature_dim = 512

    # 模拟图像特征
    image_features = [torch.randn(feature_dim) for _ in range(num_images)]

    # 模拟物体特征（每张图像有1-3个物体）
    object_features_list = []
    for i in range(num_images):
        num_objects = np.random.randint(1, 4)
        obj_features = [torch.randn(feature_dim) for _ in range(num_objects)]
        object_features_list.append(obj_features)

    # 使用特征处理
    feature_result = selector.process_images_with_features(
        image_features, object_features_list, instruction_text
    )
    print(f"选择的关键帧索引: {feature_result['keyframe_indices']}")

    print("\n✅ 测试完成!")