"""
Text Preprocessing Utilities
Tiền xử lý văn bản cho từng loại model
"""

import re
from typing import List, Tuple
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


class ViT5Preprocessor(BasePreprocessor):
    """Preprocessor cho model ViT5 thuần"""
    
    def preprocess(self, text: str, max_length: int = 1024, **kwargs) -> dict:
        # Clean text
        cleaned = clean_text(text)
        
        # Truncate nếu quá dài (ViT5 context ~ 1024 tokens)
        truncated = truncate_text(cleaned, max_chars=3000)
        
        # Thêm prefix cho ViT5
        processed = f"summarize: {truncated}"
        
        return {
            "processed_text": processed,
            "original_length": len(text),
            "processed_length": len(truncated),
            "was_truncated": len(truncated) < len(cleaned)
        }


class PhoBertViT5Preprocessor(BasePreprocessor):
    """Preprocessor cho model hybrid PhoBERT-ViT5"""
    
    def preprocess(
        self, 
        text: str, 
        top_k_ratio: float = 0.6,
        **kwargs
    ) -> dict:
        # Clean text
        cleaned = clean_text(text)
        
        # Segment thành câu
        sentences = segment_sentences(cleaned)
        
        # Limit số câu để tránh quá tải
        max_sentences = 50
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        return {
            "processed_text": cleaned,
            "sentences": sentences,
            "num_sentences": len(sentences),
            "top_k_ratio": top_k_ratio,
            "original_length": len(text)
        }


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


# Factory function
def get_preprocessor(model_type: str) -> BasePreprocessor:
    """Lấy preprocessor phù hợp với loại model"""
    preprocessors = {
        "vit5": ViT5Preprocessor(),
        "phobert_vit5": PhoBertViT5Preprocessor(),
        "qwen": QwenPreprocessor(),
    }
    
    if model_type not in preprocessors:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return preprocessors[model_type]
