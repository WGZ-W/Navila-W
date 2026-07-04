import spacy
from typing import List, Dict, Set, Tuple
from collections import Counter


class EnglishNounExtractor:
    def __init__(self, model_name="en_core_web_sm"):
        """
        初始化英文名词提取器

        参数:
            model_name: spaCy英文模型名称，默认使用 en_core_web_sm
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"模型 {model_name} 未找到，请运行: python -m spacy download {model_name}")
            raise

        # 场景相关词汇
        self.scene_keywords = {
            "location", "place", "scene", "environment", "background", "setting", "area",
            "park", "street", "room", "office", "school", "mall", "restaurant", "beach",
            "forest", "mountain", "river", "city", "countryside", "indoor", "outdoor",
            "kitchen", "bedroom", "garden", "classroom", "stadium", "theater", "hospital",
            "building", "stadium", "field", "facility", "structure", "house", "apartment",
            "shop", "store", "library", "museum", "airport", "station", "bridge", "tower",
            "monument", "landmark", "view", "landscape", "vista", "panorama"
        }

        # 物体相关词汇
        self.object_keywords = {
            "object", "item", "thing", "entity",
            "dog", "cat", "car", "ball", "table", "chair", "computer", "phone", "book",
            "flower", "tree", "house", "person", "clothes", "food", "cup", "pen", "key",
            "bag", "shoe", "hat", "glasses", "watch", "roof", "track", "border", "deer",
            "umbrella", "desk", "lamp", "window", "door", "wall", "floor", "ceiling",
            "painting", "picture", "sculpture", "statue", "fountain", "bench", "fence",
            "vehicle", "bicycle", "motorcycle", "bus", "train", "plane", "boat", "ship"
        }

        # 场景相关介词
        self.scene_prepositions = {"in", "at", "on", "inside", "outside", "under", "over", "beside", "near", "around",
                                   "between"}

        # 要排除的抽象名词
        self.abstract_nouns = {
            "color", "colors", "way", "time", "year", "work", "day", "man", "world", "life", "hand", "part",
            "child", "eye", "woman", "case", "point", "company", "fact", "idea", "thought", "feeling", "emotion",
            "love", "hate", "fear", "hope", "dream", "memory", "knowledge", "truth", "lie", "beauty", "ugliness",
            "goodness", "evil", "justice", "injustice", "freedom", "slavery", "peace", "war", "health", "disease",
            "wealth", "poverty", "success", "failure", "victory", "defeat", "strength", "weakness", "courage", "fear",
            "wisdom", "foolishness", "intelligence", "stupidity", "creativity", "boredom", "happiness", "sadness",
            "anger", "calm", "patience", "impatience", "kindness", "cruelty", "generosity", "greed", "honesty",
            "dishonesty",
            "loyalty", "betrayal", "faith", "doubt", "trust", "suspicion", "respect", "disrespect", "pride", "humility",
            "confidence", "insecurity", "optimism", "pessimism", "realism", "idealism", "pragmatism", "romanticism",
            "modernism", "traditionalism", "progress", "regress", "development", "decline", "growth", "decay",
            "creation", "destruction", "birth", "death", "beginning", "end", "start", "finish", "arrival", "departure",
            "presence", "absence", "existence", "nonexistence", "reality", "fantasy", "illusion", "delusion",
            "perception", "misperception", "understanding", "misunderstanding", "comprehension", "incomprehension",
            "awareness", "unawareness", "consciousness", "unconsciousness", "attention", "inattention", "focus",
            "distraction",
            "concentration", "dispersion", "unity", "division", "harmony", "discord", "agreement", "disagreement",
            "cooperation", "competition", "collaboration", "conflict", "peace", "war", "friendship", "enmity",
            "love", "hate", "affection", "indifference", "passion", "apathy", "enthusiasm", "boredom", "interest",
            "disinterest",
            "curiosity", "indifference", "surprise", "expectation", "disappointment", "satisfaction", "dissatisfaction",
            "pleasure", "pain", "comfort", "discomfort", "ease", "difficulty", "simplicity", "complexity", "clarity",
            "confusion",
            "order", "chaos", "organization", "disorganization", "system", "randomness", "pattern", "randomness",
            "symmetry", "asymmetry", "balance", "imbalance", "stability", "instability", "security", "insecurity",
            "safety", "danger", "risk", "certainty", "uncertainty", "probability", "improbability", "possibility",
            "impossibility",
            "necessity", "contingency", "fate", "chance", "luck", "misfortune", "fortune", "destiny", "freewill",
            "determinism", "indeterminism", "causality", "accident", "intention", "accident", "purpose", "aimlessness",
            "goal", "process", "result", "consequence", "effect", "cause", "reason", "explanation", "mystery",
            "secret", "revelation", "concealment", "truth", "falsehood", "fact", "fiction", "reality", "appearance",
            "essence", "existence", "being", "nothingness", "something", "nothing", "everything", "anything",
            "nothingness"
        }

        # 停用词（代词、连词等）
        self.stop_words = {"it", "they", "he", "she", "this", "that", "these", "those", "here", "there", "which", "who",
                           "what"}

        # 常见动词的非名词形式（可能被错误识别为名词）
        self.verb_forms = {"shift", "move", "go", "come", "walk", "run", "jump", "turn", "look", "see", "watch",
                           "listen", "hear"}

    def extract_nouns(self, instruction_text: str) -> List[Dict]:
        """
        从英文指令文本中提取名词短语，并区分场景和物体

        参数:
            instruction_text: 英文指令文本

        返回:
            nouns_list: 名词字典列表
        """
        doc = self.nlp(instruction_text)
        noun_phrases = []

        # 提取名词短语
        for chunk in doc.noun_chunks:
            if self._is_valid_noun_chunk(chunk):
                noun_type, confidence = self._classify_noun_phrase(chunk)
                if confidence >= 0.5:  # 只保留置信度较高的名词
                    noun_phrases.append({
                        "text": chunk.text,
                        "type": noun_type,
                        "confidence": confidence,
                        "root": chunk.root.text,
                        "pos": chunk.root.pos_
                    })

        # 提取命名实体
        for ent in doc.ents:
            if ent.label_ in {"LOC", "GPE", "FAC", "PRODUCT", "ORG", "PERSON", "WORK_OF_ART"}:
                noun_type, confidence = self._classify_named_entity(ent)
                if confidence >= 0.5:
                    noun_phrases.append({
                        "text": ent.text,
                        "type": noun_type,
                        "confidence": confidence,
                        "root": ent.root.text,
                        "pos": ent.root.pos_,
                        "entity_type": ent.label_
                    })

        # 处理和分析名词
        return self._process_nouns(noun_phrases, doc)

    def _is_valid_noun_chunk(self, chunk) -> bool:
        """
        检查名词短语是否有效
        """
        # 检查长度
        if len(chunk.text.split()) > 5:
            return False

        # 检查根词是否是停用词
        root_lower = chunk.root.text.lower()
        if root_lower in self.stop_words:
            return False

        # 检查是否是抽象名词
        if root_lower in self.abstract_nouns:
            return False

        # 检查是否是动词形式
        if root_lower in self.verb_forms:
            return False

        # 检查词性：必须是名词或专有名词
        if chunk.root.pos_ not in {"NOUN", "PROPN"}:
            return False

        # 检查是否有实际意义（排除单个字母等）
        if len(chunk.root.text) < 2:
            return False

        return True

    def _classify_noun_phrase(self, chunk) -> tuple:
        """
        分类名词短语为场景或物体
        """
        text_lower = chunk.text.lower()
        root_lower = chunk.root.text.lower()

        # 检查预定义关键词
        if text_lower in self.scene_keywords or root_lower in self.scene_keywords:
            return "scene", 0.9
        if text_lower in self.object_keywords or root_lower in self.object_keywords:
            return "object", 0.9

        # 检查介词上下文
        if chunk.root.head.pos_ == "ADP" and chunk.root.head.text.lower() in self.scene_prepositions:
            return "scene", 0.7

        # 基于实体类型
        if chunk.root.ent_type_:
            if chunk.root.ent_type_ in {"LOC", "GPE", "FAC"}:
                return "scene", 0.8
            if chunk.root.ent_type_ in {"PRODUCT", "PERSON", "WORK_OF_ART"}:
                return "object", 0.7
            if chunk.root.ent_type_ == "ORG":
                return "object", 0.6

        # 基于词性标注
        if chunk.root.pos_ == "PROPN":
            return "scene", 0.7

        # 基于依存关系
        if chunk.root.dep_ in {"pobj", "attr", "nsubj"}:
            return "object", 0.6

        # 默认分类
        return "object", 0.5

    def _classify_named_entity(self, entity) -> tuple:
        """
        分类命名实体
        """
        if entity.label_ in {"LOC", "GPE", "FAC"}:
            return "scene", 0.85
        elif entity.label_ in {"PRODUCT", "PERSON", "WORK_OF_ART"}:
            return "object", 0.8
        elif entity.label_ == "ORG":
            return "object", 0.7
        else:
            return "object", 0.6

    def _process_nouns(self, nouns_list: List[Dict], doc) -> List[Dict]:
        """
        处理名词列表：去重、过滤、排序
        """
        if not nouns_list:
            return []

        # 第一步：基于文本的去重
        text_to_noun = {}
        for noun in nouns_list:
            text = noun["text"].lower()
            if text not in text_to_noun:
                text_to_noun[text] = noun
            elif noun["confidence"] > text_to_noun[text]["confidence"]:
                text_to_noun[text] = noun

        # 第二步：基于根词的去重（对于相似短语）
        root_to_noun = {}
        for noun in text_to_noun.values():
            root = noun["root"].lower()
            # 跳过抽象名词
            if root in self.abstract_nouns:
                continue

            if root not in root_to_noun:
                root_to_noun[root] = noun
            else:
                # 保留更长或置信度更高的短语
                current_noun = root_to_noun[root]
                if (noun["confidence"] > current_noun["confidence"] or
                        (noun["confidence"] == current_noun["confidence"] and
                         len(noun["text"]) > len(current_noun["text"]))):
                    root_to_noun[root] = noun

        # 转换为列表
        result = list(root_to_noun.values())

        # 第三步：进一步过滤
        result = [noun for noun in result if self._should_keep_noun(noun)]

        # 第四步：按置信度排序
        result.sort(key=lambda x: x["confidence"], reverse=True)

        return result

    def _should_keep_noun(self, noun: Dict) -> bool:
        """
        检查是否应该保留这个名词
        """
        # 检查置信度
        if noun["confidence"] < 0.5:
            return False

        # 检查是否是抽象名词
        if noun["root"].lower() in self.abstract_nouns:
            return False

        # 检查是否是停用词
        if noun["root"].lower() in self.stop_words:
            return False

        # 检查是否是动词形式
        if noun["root"].lower() in self.verb_forms:
            return False

        # 检查是否有实际内容
        if len(noun["text"].strip()) < 3:
            return False

        # 检查是否包含无意义的词
        meaningless_words = {"color", "colors", "shift", "afterwards"}
        for word in meaningless_words:
            if word in noun["text"].lower():
                return False

        return True

    def extract_scene_objects(self, instruction_text: str) -> Dict[str, List[str]]:
        """
        提取场景和物体名词的简化版本

        参数:
            instruction_text: 英文指令文本

        返回:
            包含场景和物体名词的字典
        """
        nouns = self.extract_nouns(instruction_text)

        scene_nouns = [noun["text"] for noun in nouns if noun["type"] == "scene"]
        object_nouns = [noun["text"] for noun in nouns if noun["type"] == "object"]

        return {
            "scenes": list(set(scene_nouns)),
            "objects": list(set(object_nouns)),
            "all_nouns": [noun["text"] for noun in nouns]
        }

    def get_noun_statistics(self, instruction_text: str) -> Dict:
        """
        获取名词提取的统计信息

        参数:
            instruction_text: 英文指令文本

        返回:
            统计信息字典
        """
        nouns = self.extract_nouns(instruction_text)
        scene_nouns = [noun for noun in nouns if noun["type"] == "scene"]
        object_nouns = [noun for noun in nouns if noun["type"] == "object"]

        return {
            "total_nouns": len(nouns),
            "scene_nouns_count": len(scene_nouns),
            "object_nouns_count": len(object_nouns),
            "scene_nouns": [noun["text"] for noun in scene_nouns],
            "object_nouns": [noun["text"] for noun in object_nouns],
            "confidence_summary": {
                "average": sum(n["confidence"] for n in nouns) / len(nouns) if nouns else 0,
                "min": min(n["confidence"] for n in nouns) if nouns else 0,
                "max": max(n["confidence"] for n in nouns) if nouns else 0
            }
        }


# 使用示例
if __name__ == "__main__":
    # 创建英文名词提取器
    extractor = EnglishNounExtractor()

    # 测试文本
    test_texts = [
        "A dog is chasing a ball in the park",
        "There is a computer and a phone on the office desk",
        "People are holding umbrellas on the rainy street",
        "Advance forward to a large gray building with a rectangular flat roof; then, slightly turn left slightly and proceed to it. Afterwards, shift left towards a large stadium encompassed by green and brown colors, highlighting an open field with a surrounding sports track. Walk straight to a small sports field, green in color, with a rectangular layout and a red border. Finally, slightly turn left and move ahead to a medium-sized sports facility distinguished by a curved white and beige roof structure."
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n{'=' * 60}")
        print(f"Test {i}:")
        print(f"Input: {text}")
        print("-" * 50)

        # 方法1: 获取详细名词信息
        nouns = extractor.extract_nouns(text)
        print("Detailed extraction:")
        for noun in nouns:
            print(f"  - {noun['text']} ({noun['type']}, confidence: {noun['confidence']:.2f})")

        # 方法2: 获取分类好的名词列表
        categorized = extractor.extract_scene_objects(text)
        print(f"\nScene nouns: {categorized['scenes']}")
        print(f"Object nouns: {categorized['objects']}")

        # 统计信息
        stats = extractor.get_noun_statistics(text)
        print(f"\nStatistics:")
        print(f"  Total nouns: {stats['total_nouns']}")
        print(f"  Scene nouns: {stats['scene_nouns_count']}")
        print(f"  Object nouns: {stats['object_nouns_count']}")

        # 特别针对长文本测试
        if i == 4:
            print(f"\n{'=' * 60}")
            print("Additional analysis for long text:")
            print(f"All extracted nouns: {categorized['all_nouns']}")