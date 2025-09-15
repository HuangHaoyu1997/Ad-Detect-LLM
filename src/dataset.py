from calendar import c
import os
import json
import re
import random
from pathlib import Path
from typing import List, Tuple, Dict

def reverse_chunk_split(context: str, min_length: int, max_length: int):
    """
    从字符串末尾切一个段，长度在 [min_length, max_length] 之间随机，尽可能在句子的分隔符处切分。
    """
    if not context:
        return ''

    split_chars = ['。', '！', '？', '.', '!', '?']

    # 随机一个切片长度
    max_chunk_len = min(max_length, len(context))
    min_chunk_len = min(min_length, max_chunk_len)
    rand_len = random.randint(min_chunk_len, max_chunk_len)

    start_pos = max(0, len(context) - rand_len)

    # 在 [start_pos, 末尾) 之间从后向前找分隔符
    best_cut = None
    for i in range(len(context) - 1, start_pos - 1, -1):
        if context[i] in split_chars:
            best_cut = i + 1  # 在分隔符之后切
            break

    if best_cut and best_cut != len(context):
        return context[best_cut:]
    else:
        return context[start_pos:]

def parse_txt_file(file_path: str) -> Dict:
    """
    解析单个txt文件，提取视频标题、正文内容和广告内容
    
    Args:
        file_path: txt文件路径
        
    Returns:
        dict: 包含标题、正文段落、广告段落、全文
        - 'title': video_title,
        - 'text_segments': text_segments,
        - 'ad_segments': ad_segments,
        - 'full_content': content
    """
    # 从文件名提取视频标题
    video_title = Path(file_path).stem
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按行分割内容
    lines = content.strip().split('\n')
    
    text_segments = []  # 正文段落
    ad_segments = []    # 广告段落
    current_segment = []
    is_ad = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 检查是否是广告开始
        if line.startswith('- '):
            # 保存之前的段落
            if current_segment:
                if is_ad:
                    ad_segments.append('\n'.join(current_segment))
                else:
                    text_segments.append('\n'.join(current_segment))
                current_segment = []
            
            # 开始新的广告段落
            is_ad = True
            current_segment.append(line[2:])  # 移除'- '前缀
        else:
            # 如果之前是广告，现在不是，说明广告结束
            if is_ad and not line.startswith('- '):
                if current_segment:
                    ad_segments.append('\n'.join(current_segment))
                    current_segment = []
                is_ad = False
            
            current_segment.append(line)
    
    # 处理最后一个段落
    if current_segment:
        if is_ad:
            ad_segments.append('\n'.join(current_segment))
        else:
            text_segments.append('\n'.join(current_segment))
    
    return {
        'title': video_title,
        'text_segments': text_segments,
        'ad_segments': ad_segments,
        'full_content': content
    }

def create_system_prompt() -> str:
    """创建系统提示"""

    return """你是一个专业的视频内容分析助手。你的任务是判断给定的文本片段是否为广告内容。

广告内容通常具有以下特征：
1. 包含商品推广、品牌宣传信息
2. 包含购买链接、优惠信息、促销活动
3. 包含"点击链接"、"立即购买"、"限时优惠"等营销用语
4. 与视频主题内容无关或关联性较弱

请根据视频标题和上下文内容，判断待分析文本是否为广告。只回答"是"或"否"。"""

def create_user_prompt(video_title: str, context: str, text: str) -> str:
    """创建用户提示"""

    return f"""视频标题：{video_title}

上下文内容：
{context}

待分析文本：
{text}

请判断待分析文本是否为广告内容："""
    
class SFTDatasetBuilder:
    def __init__(self, txt_folder_path: str, output_path: str = "./sft_dataset"):
        """
        初始化数据集构建器
        
        Args:
            txt_folder_path: 包含txt文件的文件夹路径
            output_path: 输出数据集的路径
        """
        self.txt_folder_path = txt_folder_path
        self.output_path = output_path
        self.context_min_length = 50  # context最小长度
        self.context_max_length = 200  # context最大长度
        self.text_min_length = 50  # text最小长度
        self.text_max_length = 200  # text最大长度
        self.positive_negative_ratio = 1.0  # 正负样本比例 1:1
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
    def get_context_before_position(self, full_text: str, position: int) -> str:
        """
        获取指定位置之前的上下文，保持文本连贯性
        
        Args:
            full_text: 完整文本
            position: 目标位置
            
        Returns:
            上下文文本
        """
        if position <= 0:
            return ""
        
        # 获取position之前的文本
        before_text = full_text[:position].strip()
        
        if len(before_text) <= self.context_min_length:
            return before_text
        
        # 如果超过最大长度，从句号处截断
        if len(before_text) > self.context_max_length:
            # 找到最后一个句号的位置
            sentences = re.split(r'[。！？]', before_text)
            if len(sentences) > 1:
                # 保留到倒数第二个句子的结尾
                truncated = '。'.join(sentences[:-1])
                if len(truncated) >= self.context_min_length:
                    return truncated + '。'
            
            # 如果没有合适的句号，直接截断到最大长度
            return before_text[-self.context_max_length:]
        
        return before_text
    
    def split_text_into_chunks(self, text: str, max_length: int, min_length: int) -> List[str]:
        """
        将文本分割成合适长度的块，保持完整性
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        if len(text) <= max_length:
            return [text] if len(text) >= min_length else []
        
        # len(text) > max_length
        chunks = []
        sentences = re.split(r'([。！？])', text)
        
        current_chunk = ""
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]
            
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence
            else:
                if len(current_chunk) >= min_length:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if len(current_chunk) >= min_length:
            chunks.append(current_chunk)
        
        return chunks
    

    
    def generate_samples_from_file(self, file_data: Dict) -> List[Dict]:
        """
        从单个文件数据生成训练样本
        
        Args:
            file_data: 解析后的文件数据
            
        Returns:
            样本列表
        """
        samples = []
        full_content = file_data['full_content']
        print(type(full_content))
        title = file_data['title']
        # 生成正样本（广告）
        for ad_segment in file_data['ad_segments']: # file_data['ad_segments']是list，一般而言只有一个item
            # 找到广告的开头在全文中的位置
            ad_position = full_content.find(ad_segment)
            if ad_position == -1:
                continue
            
            # 获取广告之前的上下文
            context = self.get_context_before_position(full_content, ad_position)
            # 分割正文文本
            context_chunks = self.split_text_into_chunks(context, self.context_max_length, self.context_min_length)
            
            # 分割广告文本
            ad_chunks = self.split_text_into_chunks(ad_segment, self.text_max_length, self.text_min_length)
            for chunk_context in context_chunks:
                for chunk_ad in ad_chunks:
                    sample = {
                        "messages": [
                            {"role": "system", "content": create_system_prompt()},
                            {"role": "user", "content": create_user_prompt(title, chunk_context, chunk_ad)},
                            {"role": "assistant", "content": "是"}
                        ]
                    }
                    samples.append(sample)
        
        # 生成负样本（非广告）
        negative_count_needed = int(len([s for s in samples]) * self.positive_negative_ratio)
        negative_samples = []
        
        for text_segment in file_data['text_segments']:
            # 找到正文在全文中的位置
            text_position = full_content.find(text_segment)
            if text_position == -1:
                continue
            
            # 分割正文文本
            text_chunks = self.split_text_into_chunks(text_segment, self.text_max_length, self.text_min_length)
            
            for chunk in text_chunks:
                # 如果负样本数量已经达到要求，则停止循环，防止生成过多的负样本
                if len(negative_samples) >= negative_count_needed: 
                    break
                
                # 找到这个chunk在全文中的位置
                chunk_position = full_content.find(chunk, text_position)
                if chunk_position == -1:
                    continue
                
                context = self.get_context_before_position(full_content, chunk_position)
                # context出现在chunk前面，在[self.text_max_length, self.text_min_length]的范围内随机截取context的长度
                context = reverse_chunk_split(context, self.text_min_length, self.text_max_length)
                print(len(context))
                sample = {
                    "messages": [
                        {"role": "system", "content": create_system_prompt()},
                        {"role": "user", "content": create_user_prompt(title, context, chunk)},
                        {"role": "assistant", "content": "否"}
                    ]
                }
                negative_samples.append(sample)
            
            if len(negative_samples) >= negative_count_needed:
                break
        
        samples.extend(negative_samples)
        return samples
    
    def build_dataset(self, train_ratio: float = 0.8):
        """
        构建完整数据集
        
        Args:
            train_ratio: 训练集占全部txt文件的比例
        """
        # 获取所有txt文件
        txt_files = list(Path(self.txt_folder_path).glob("*.txt"))
        
        if not txt_files:
            raise ValueError(f"在 {self.txt_folder_path} 中未找到txt文件")
        # print(f"找到 {len(txt_files)} 个txt文件")
        
        # 按文件大小排序
        txt_files.sort(key=lambda x: x.stat().st_size)
        
        # 划分训练集和测试集
        split_index = int(len(txt_files) * train_ratio)
        train_files = txt_files[:split_index]
        test_files = txt_files[split_index:]
        
        print(f"训练集文件数: {len(train_files)}")
        print(f"测试集文件数: {len(test_files)}")
        
        # 生成训练集
        train_samples = []
        for file_path in train_files:
            print(f"处理训练文件: {file_path.name}")
            try:
                file_data = parse_txt_file(str(file_path))
                samples = self.generate_samples_from_file(file_data)
                train_samples.extend(samples)
                print(f"  生成样本数: {len(samples)}")
            except Exception as e:
                print(f"  处理文件 {file_path.name} 时出错: {e}")
        
        # 生成测试集
        test_samples = []
        for file_path in test_files:
            print(f"处理测试文件: {file_path.name}")
            try:
                file_data = parse_txt_file(str(file_path))
                samples = self.generate_samples_from_file(file_data)
                test_samples.extend(samples)
                print(f"  生成样本数: {len(samples)}")
            except Exception as e:
                print(f"  处理文件 {file_path.name} 时出错: {e}")
        
        # 打乱样本顺序
        random.shuffle(train_samples)
        random.shuffle(test_samples)
        
        # 保存数据集
        train_path = os.path.join(self.output_path, "train.json")
        test_path = os.path.join(self.output_path, "test.json")
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
        
        # 统计信息
        train_positive = sum(1 for s in train_samples if s["messages"][2]["content"] == "是")
        train_negative = len(train_samples) - train_positive
        test_positive = sum(1 for s in test_samples if s["messages"][2]["content"] == "是")
        test_negative = len(test_samples) - test_positive
        
        print("\n=== 数据集构建完成 ===")
        print(f"训练集总样本数: {len(train_samples)}")
        print(f"  正样本(广告): {train_positive}")
        print(f"  负样本(非广告): {train_negative}")
        print(f"测试集总样本数: {len(test_samples)}")
        print(f"  正样本(广告): {test_positive}")
        print(f"  负样本(非广告): {test_negative}")
        print(f"训练集保存至: {train_path}")
        print(f"测试集保存至: {test_path}")

if __name__ == '__main__':
    # 配置参数
    txt_folder_path = "E:/LLM/Ad-Detect-LLM/done/"  # 替换为你的txt文件文件夹路径
    output_path = "./sft_dataset"    # 输出路径

    # 创建数据集构建器
    builder = SFTDatasetBuilder(txt_folder_path, output_path)

    # 可调整的参数
    builder.context_min_length = 50    # context最小长度
    builder.context_max_length = 200   # context最大长度
    builder.text_min_length = 30        # text最小长度
    builder.text_max_length = 100       # text最大长度
    builder.positive_negative_ratio = 1.0  # 正负样本比例

    # 构建数据集
    builder.build_dataset(train_ratio=0.8)