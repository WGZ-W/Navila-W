import json
import spacy
import pandas as pd
import re
from collections import Counter

# 加载 spaCy 模型（需要先下载：python -m spacy download en_core_web_sm）
nlp = spacy.load("en_core_web_sm")


def extract_noun_phrases(text):
    """提取文本中的所有名词短语（NP）"""
    doc = nlp(text)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        # 过滤掉太短（少于2个词）或纯代词的名词短语
        if len(chunk.text.split()) >= 2 and chunk.root.pos_ not in ("PRON", "DET"):
            # 清理标点并标准化空格
            phrase = re.sub(r'\s+', ' ', chunk.text.strip()).strip()
            if phrase:
                noun_phrases.append(phrase)
    return noun_phrases


def classify_noun_phrase(phrase):
    """
    根据短语中的关键词判断类别：
    0: 物体级实体 (object) - 小物体、部件、细节
    1: 场景级地标 (scene) - 建筑、区域、大型结构
    """
    phrase_lower = phrase.lower()

    # 明确指示场景级的关键词（通常是大型建筑或区域）
    scene_keywords = [
        "building", "skyscraper", "high-rise", "tower", "structure", "complex",
        "facade", "exterior", "rooftop", "roof", "floor", "street", "square",
        "plaza", "bridge", "crane", "billboard", "antenna", "tower crane",
        "high rise", "apartment building", "office building", "commercial building",
        "residential building", "landmark", "skyline", "cityscape", "block",
        "wall", "concrete wall", "brick wall", "pavement", "road", "intersection"
    ]

    # 明确指示物体级的关键词（小部件、细节物体）
    object_keywords = [
        "window", "door", "balcony", "vent", "pipe", "air conditioning unit",
        "hvac", "grille", "louver", "rail", "stair", "ladder", "column",
        "pillar", "beam", "crack", "stain", "mark", "shadow", "light",
        "reflection", "sign", "advertisement", "logo", "text", "illustration",
        "statue", "sculpture", "clock", "spire", "dome", "chimney", "satellite dish",
        "antenna", "crane hook", "hook", "scaffolding", "scaffold", "frame",
        "girder", "truss", "cable", "wire", "pipe", "duct", "outlet", "inlet"
    ]

    for kw in scene_keywords:
        if kw in phrase_lower:
            return 1
    for kw in object_keywords:
        if kw in phrase_lower:
            return 0

    # 启发式：如果短语中包含 "building" 但前面有 "small", "medium" 等，不一定为 scene，
    # 但上面已经优先匹配了 building -> 1。这里再处理 ambiguous 情况：
    # 如果短语长度较短（<=2词）且没有明显场景词，倾向物体级
    if len(phrase.split()) <= 2:
        return 0
    # 默认：如果描述的是某个具体部位（如 "the gray wall"）可能是物体，但墙也算场景？通常墙属于建筑的一部分，可以视为场景地标。
    # 更精细的可以再调整，这里简单地将包含 "wall", "facade", "roof" 等归为场景。
    if any(w in phrase_lower for w in ["wall", "facade", "roof", "surface"]):
        return 1
    # 剩余情况：默认为物体级（保守）
    return 0


def main():
    # 读取 eval.json
    with open("eval.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for entry in data:
        instruction = entry.get("gpt_instruction", "")
        if not instruction:
            continue

        # 提取名词短语
        noun_phrases = extract_noun_phrases(instruction)
        # 去重（保留顺序）
        seen = set()
        unique_phrases = []
        for phrase in noun_phrases:
            if phrase not in seen:
                seen.add(phrase)
                unique_phrases.append(phrase)

        for phrase in unique_phrases:
            label = classify_noun_phrase(phrase)
            records.append({"text": phrase, "label": label})

    # 保存为 CSV
    df = pd.DataFrame(records)
    df.to_csv("noun_phrases.csv", index=False, encoding="utf-8")
    print(f"生成 CSV 文件，共 {len(df)} 个名词短语")
    print("\n类别分布：")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()