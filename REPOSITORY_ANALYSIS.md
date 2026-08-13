# NaVILA Repository Analysis

## 总体结论

这是一个以 NaVILA 为核心的视觉语言动作（VLA）导航研究仓库，主要链路如下：

```text
视频/环境数据
    -> 关键帧选择、目标检测、名词提取
    -> SigLIP/CLIP视觉编码
    -> LLaVA/VILA多模态语言模型
    -> ActionTokenizer动作离散化
    -> 训练或机器人导航推理
    -> Habitat/VLN-CE评测
```

当前工作目录约 13 GB，实际文件约 2077 个；Git 已跟踪约 1281 个文件。仓库包含大量模型、仿真环境和分析图片，属于研究实验型仓库，而不是轻量级 Python 包。

## 目录结构

### `llava/`

核心模块，约 991 个文件。

- `llava/model/`：LLaMA、Mistral、Gemma、MPT 等语言模型；SigLIP、CLIP、Intern、Radio 等视觉编码器；多模态投影层；`history_mamba/` 和 `history_transformer/` 历史状态建模；`action_tokenizer.py` 动作 token 化。
- `llava/data/`、`llava/dataset/`：数据集、数据混合、视频和图像预处理。
- `llava/train/`：训练器、DeepSpeed、序列并行、LoRA 和长上下文支持。
- `llava/trl/`：SFT、DPO、PPO、奖励模型等训练扩展。
- `llava/eval.py`、`eval_motion.py`：导航动作预测和运动分析。
- `llava/cli/`：`run.py` 提交 SLURM 任务，`eval.py` 批量运行评测。

### `evaluation/`

独立的 Habitat/VLN-CE 导航评测系统。`evaluation/run.py` 是训练、评测、推理统一入口；`habitat_extensions/` 提供传感器、地图、规划器和任务扩展；`vlnce_baselines/` 提供导航策略、训练器、模型及 R2R/RxR 配置；`scripts/eval/r2r.sh` 负责多 GPU R2R 评测；`scripts/eval_jsons.py` 汇总结果。完整数据集和场景需要外部下载到 `evaluation/data`。

### `keyframe/`

关键帧和视觉语义预处理：`clip_wrapper.py` 封装 CLIP/SigLIP，`object_detector.py` 使用 YOLOv5/YOLOv8，`noun_extractor.py` 用 spaCy 提取英文名词，`keyframe_selector.py` 根据场景、物体和文本相似度选择关键帧，`simple_keyframe_selector.py` 提供简化版本。

这里同时存在 `keyframe/` 和 `keyframe/keyframe/` 两套近似重复实现，可能是历史迁移或兼容代码，维护时应确认实际导入路径。

### `VideoMamba/`

独立的视频 Mamba 子项目，约 278 个文件。`mamba/` 包含 Selective Scan、CUDA kernel 和 Mamba SSM；`causal-conv1d/` 是 CUDA 因果卷积扩展；`videomamba/video_sm/` 面向视频分类与预训练；`videomamba/video_mm/` 面向视频检索、VQA 和多模态任务。大量 `run*.sh` 属于原始实验配置，不一定在 NaVILA 主链路中使用。

### 其他目录和根目录脚本

- `envs/`：AirSim、Gazebo、Unreal 场景资源，当前包含大型二进制文件。
- `configs/`：环境 YAML、评测 JSON、名词短语脚本和词表。
- `scripts/`：`train/sft_8frames.sh` 监督微调，`fsdp_open.sh` 和 `zero3.json` 分布式训练，`extract_rawframes.py` 视频抽帧。
- 根目录测试和诊断脚本：`test*.py`、`data_test*.py`、`debug_*.py`、`check_tfrecords.py`、`diagnose_tfrecord.py`、`calculate_results.py`、`train_bert_classifier.py`。

## 模型和数据资产

本地资产包括：`bert/`（约 2.8 GB）、`siglip-base-patch16-224/`（约 1.6 GB）、`bert_scene_object_classifier.pt`（约 439 MB）、多个 `yolov8n.pt` 副本，以及 `llava/eval_vis/` 中约 666 个轨迹和动作可视化图片。这些文件多数被 Git 忽略，适合作为本地运行资产，不适合继续提交到源码仓库。

## 构建和运行方式

项目使用 `pyproject.toml`，依赖 PyTorch 2.3、Transformers 4.37.2、DeepSpeed、FlashAttention、Habitat/VLN-CE 等，GPU 和版本要求较严格。

```bash
./environment_setup.sh navila
conda activate navila
pip install -e .
pip install -e ".[train]"
pip install -e "[eval]"
```

训练：

```bash
bash scripts/train/sft_8frames.sh
```

单 GPU R2R 评测：

```bash
cd evaluation
bash scripts/eval/r2r.sh CKPT_PATH 1 0 "0"
```

## 当前维护风险

1. 环境依赖敏感：Habitat、FlashAttention、CUDA、TensorFlow、DeepSpeed 和 Transformers 版本耦合明显。
2. 数据路径依赖强：训练路径在 `llava/data/datasets_mixture.py`，评测数据和场景要求放在 `evaluation/data`。
3. 正式代码、临时调试代码和测试代码混在根目录，增加理解成本。
4. 存在重复实现和模型文件：`keyframe/keyframe/` 与 `keyframe/` 重复，`yolov8n.pt` 多处复制。
5. `llava/cli/eval.py` 的并发进程调度逻辑中 `task` 变量使用位置可疑，实际运行前应重点验证。
6. Git 工作区当前不干净，存在 `.gitignore` 修改、未跟踪文件和本次生成的 `AGENTS.md`，应与源码分析区分处理。
