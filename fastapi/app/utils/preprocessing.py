"""
Text Preprocessing Utilities
Tiền xử lý văn bản cho từng loại model
"""

import re
from typing import List
from abc import ABC, abstractmethod


def clean_text(text: str) -> str:
    """
    Làm sạch văn bản cơ bản:
    - Chuẩn hóa khoảng trắng
    - Loại bỏ ký tự đặc biệt không cần thiết
    """
    # Chuẩn hóa xuống dòng
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Loại bỏ khoảng trắng đầu/cuối mỗi dòng
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    
    return text.strip()


def segment_sentences(text: str) -> List[str]:
    """
    Tách văn bản thành các câu.
    Sử dụng regex đơn giản (có thể thay bằng underthesea nếu cần)
    """
    # Pattern cho dấu kết thúc câu tiếng Việt
    sentence_endings = r'(?<=[.!?])\s+'
    
    # Tách câu
    sentences = re.split(sentence_endings, text)
    
    # Lọc câu rỗng và quá ngắn
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    
    return sentences


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Cắt văn bản nếu quá dài"""
    if len(text) <= max_chars:
        return text
    
    # Cắt tại vị trí kết thúc câu gần nhất
    truncated = text[:max_chars]
    last_period = max(
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?')
    )
    
    if last_period > max_chars * 0.5:
        return truncated[:last_period + 1]
    
    return truncated


class BasePreprocessor(ABC):
    """Base class cho preprocessor"""
    
    @abstractmethod
    def preprocess(self, text: str, **kwargs) -> dict:
        """
        Tiền xử lý văn bản.
        
        Returns:
            Dict chứa text đã xử lý và metadata
        """
        pass


class QwenPreprocessor(BasePreprocessor):
    """Preprocessor cho model Qwen2.5-3B"""
    
    SYSTEM_PROMPT = "Bạn là trợ lý tóm tắt văn bản tiếng Việt. Hãy tóm tắt ngắn gọn, giữ lại các ý chính quan trọng nhất."
    
    def preprocess(
        self, 
        text: str, 
        max_context: int = 6000,
        **kwargs
    ) -> dict:
        # Clean text
        cleaned = clean_text(text)
        
        # Truncate cho context window của Qwen
        truncated = truncate_text(cleaned, max_chars=max_context)
        
        # Build user prompt
        user_prompt = f"Tóm tắt văn bản sau:\n\n{truncated}"
        
        return {
            "processed_text": truncated,
            "system_prompt": self.SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "original_length": len(text),
            "processed_length": len(truncated),
            "was_truncated": len(truncated) < len(cleaned)
        }


class ViT5FinPreprocessor(BasePreprocessor):
    """Preprocessor cho model ViT5 Financial v2 (tncinh/vit5-financial-summarization-v2)"""
    
    def preprocess(self, text: str, max_length: int = 1024, **kwargs) -> dict:
        cleaned = clean_text(text)
        
        # Không truncate quá ngắn vì Colab server có hierarchical chunking cho text dài.
        # Chỉ giới hạn ở mức an toàn để tránh payload quá lớn.
        truncated = truncate_text(cleaned, max_chars=8000)
        
        return {
            "processed_text": truncated,
            "original_length": len(text),
            "processed_length": len(truncated),
            "was_truncated": len(truncated) < len(cleaned)
        }


class PhoBertFinancePreprocessor(BasePreprocessor):
    """Preprocessor cho model PhoBERT Finance Extractive (duong2110/phobert_finance_vi)"""
    
    def preprocess(
        self, 
        text: str, 
        **kwargs
    ) -> dict:
        # Clean text
        cleaned = clean_text(text)
        
        # Segment thành câu
        sentences = segment_sentences(cleaned)
        
        # Limit số câu (model max_sentences = 60)
        max_sentences = 60
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        return {
            "processed_text": cleaned,
            "sentences": sentences,
            "num_sentences": len(sentences),
            "original_length": len(text),
            "model_type": "extractive"
        }


# Factory function
def get_preprocessor(model_type: str) -> BasePreprocessor:
    """Lấy preprocessor phù hợp với loại model"""
    preprocessors = {
        "vit5_fin": ViT5FinPreprocessor(),
        "qwen": QwenPreprocessor(),
        "phobert_finance": PhoBertFinancePreprocessor(),
    }
    
    if model_type not in preprocessors:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return preprocessors[model_type]
