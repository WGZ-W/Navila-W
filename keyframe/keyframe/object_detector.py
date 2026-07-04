import os

import cv2
import numpy as np
import torch
from typing import List, Optional, Union
import time


class YOLODetector:
    def __init__(self,
                 model_type: str = "yolov5",
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """
        初始化YOLO检测器
        """
        self.model_type = model_type.lower()
        self.device = self._get_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # COCO类别
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        self.model = None
        self._load_model(model_path)

    def _get_device(self, device_str: str) -> torch.device:
        """获取设备"""
        if device_str.lower() == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self, model_path: Optional[str]):
        """加载模型"""
        if self.model_type == "yolov8":
            try:
                from ultralytics import YOLO
                model_path = model_path or 'yolov8n.pt'
                self.model = YOLO(model_path)
                self.input_size = 640
                print(f"Loaded YOLOv8 model: {model_path}")
            except ImportError:
                print("Install ultralytics: pip install ultralytics")
                raise

        elif self.model_type == "yolov5":
            try:
                import yolov5
                model_path = model_path or 'yolov5s.pt'
                self.model = yolov5.load(model_path, device=self.device)
                self.input_size = 640
                print(f"Loaded YOLOv5 model: {model_path}")
            except ImportError:
                try:
                    print("Trying torch hub for YOLOv5...")
                    self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                    self.input_size = 640
                except Exception as e:
                    print(f"Load failed: {e}")
                    raise

        self.model.eval()
        self.model.to(self.device)

    def detect(self, image: Union[np.ndarray, str]) -> List[dict]:
        """
        检测图像中的物体

        返回: [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'class_id': int, 'class_name': str}, ...]
        """
        # 加载图像
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot read image: {image}")

        # 转换颜色空间
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image[:, :, 0].mean() > image[:, :, 2].mean():
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        original_h, original_w = image.shape[:2]

        # 使用模型自带的推理
        if self.model_type == "yolov8":
            results = self.model(image)
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())

                        if conf >= self.conf_threshold:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class_id': cls_id,
                                'class_name': self.class_names[cls_id] if cls_id < len(
                                    self.class_names) else f'class_{cls_id}'
                            })

        else:  # yolov5
            results = self.model(image)
            detections = []

            for *xyxy, conf, cls in results.xyxy[0]:
                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls_id = int(cls)
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': cls_id,
                        'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'
                    })

        # NMS过滤
        detections = self._nms_filter(detections)
        return detections

    def _nms_filter(self, detections: List[dict]) -> List[dict]:
        """非极大值抑制过滤"""
        if not detections:
            return []

        detections.sort(key=lambda x: x['confidence'], reverse=True)

        filtered = []
        while detections:
            best = detections.pop(0)
            filtered.append(best)

            i = 0
            while i < len(detections):
                iou = self._calculate_iou(best['bbox'], detections[i]['bbox'])
                if iou > self.iou_threshold:
                    detections.pop(i)
                else:
                    i += 1

        return filtered

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """计算IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """绘制检测结果"""
        result = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']}: {det['confidence']:.2f}"

            # 随机但稳定的颜色
            color = self._get_color(det['class_id'])

            # 绘制边界框
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # 绘制标签背景
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(result, (x1, y1 - text_h - 10),
                          (x1 + text_w, y1), color, -1)

            # 绘制标签文本
            cv2.putText(result, label, (x1, y1 - 5),
                        font, font_scale, (255, 255, 255), thickness)

        return result

    def _get_color(self, class_id: int) -> tuple:
        """获取类别颜色"""
        # 使用固定颜色映射
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
        ]
        return colors[class_id % len(colors)]


# 使用示例
if __name__ == "__main__":
    # 创建检测器
    detector = YOLODetector(model_type="yolov8", device="cpu", conf_threshold=0.25)

    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image_path = "2.png"

    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误：图像文件 '{image_path}' 不存在！")
        exit(1)

    # 检测
    detections = detector.detect(image_path)

    print(f"Detected {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"  {i + 1}. {det['class_name']}: {det['confidence']:.2f}")

    # 读取图像用于绘制结果
    image = cv2.imread(image_path)

    # 绘制结果
    result_image = detector.draw_detections(image, detections)

    # 保存结果
    cv2.imwrite("detection_result.jpg", result_image)
    print("Result saved to detection_result.jpg")