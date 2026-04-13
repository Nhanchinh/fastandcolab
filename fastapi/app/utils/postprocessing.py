"""
Text Postprocessing Utilities
Hậu xử lý kết quả tóm tắt từ các model
"""

import re
from abc import ABC, abstractmethod


def clean_output(text: str) -> str:
    """
    Làm sạch output cơ bản:
    - Chuẩn hóa khoảng trắng
    - Loại bỏ ký tự thừa
    """
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    
    # Loại bỏ khoảng trắng đầu/cuối
    text = text.strip()
    
    # Đảm bảo kết thúc bằng dấu chấm
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text


def remove_repetition(text: str, min_repeat_length: int = 10) -> str:
    """
    Loại bỏ các đoạn văn bản lặp lại.
    Thường xảy ra với các model generative.
    """
    words = text.split()
    result = []
    seen_phrases = set()
    
    i = 0
    while i < len(words):
        # Kiểm tra cụm từ có lặp không
        phrase = ' '.join(words[i:i+5])  # Window 5 từ
        
        if phrase in seen_phrases and len(phrase) >= min_repeat_length:
            # Bỏ qua cụm từ lặp
            i += 5
            continue
        
        seen_phrases.add(phrase)
        result.append(words[i])
        i += 1
    
    return ' '.join(result)


def ensure_complete_sentences(text: str) -> str:
    """
    Đảm bảo text kết thúc bằng câu hoàn chỉnh.
    Cắt bỏ câu bị cắt dở nếu có.
    """
    # Tìm vị trí dấu kết thúc câu cuối cùng
    last_end = max(
        text.rfind('.'),
        text.rfind('!'),
        text.rfind('?')
    )
    
    if last_end > len(text) * 0.5:
        return text[:last_end + 1]
    
    return text


class BasePostprocessor(ABC):
    """Base class cho postprocessor"""
    
    @abstractmethod
    def postprocess(self, summary: str, **kwargs) -> dict:
        """
        Hậu xử lý kết quả tóm tắt.
        
        Returns:
            Dict chứa summary đã xử lý và metadata
        """
        pass


class QwenPostprocessor(BasePostprocessor):
    """Postprocessor cho model Qwen"""
    
    def postprocess(self, summary: str, **kwargs) -> dict:
        # Clean basic
        cleaned = clean_output(summary)
        
        # Qwen có thể trả về với format chat, cần extract phần summary
        # Loại bỏ các prefix không cần thiết
        prefixes_to_remove = [
            "Dưới đây là bản tóm tắt:",
            "Tóm tắt:",
            "Bản tóm tắt:",
            "Summary:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove repetition (LLM hay lặp)
        cleaned = remove_repetition(cleaned)
        
        # Ensure complete sentences
        cleaned = ensure_complete_sentences(cleaned)
        
        # Loại bỏ hallucination patterns phổ biến
        hallucination_patterns = [
            r'\(nguồn:.*?\)',  # (nguồn: ...)
            r'\[.*?\]',        # [...]
            r'Theo.*?cho biết,',  # Theo ... cho biết,
        ]
        
        for pattern in hallucination_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean lại sau khi loại bỏ
        cleaned = clean_output(cleaned)
        
        return {
            "summary": cleaned,
            "original_length": len(summary),
            "processed_length": len(cleaned)
        }


class ViT5FinPostprocessor(BasePostprocessor):
    """Postprocessor cho model ViT5 Financial v2"""
    
    def postprocess(self, summary: str, **kwargs) -> dict:
        # Clean basic
        cleaned = clean_output(summary)
        
        # Remove repetition (ViT5 đôi khi lặp cụm từ)
        cleaned = remove_repetition(cleaned)
        
        # Ensure complete sentences
        cleaned = ensure_complete_sentences(cleaned)
        
        return {
            "summary": cleaned,
            "original_length": len(summary),
            "processed_length": len(cleaned)
        }


class PhoBertFinancePostprocessor(BasePostprocessor):
    """Postprocessor cho model PhoBERT Finance Extractive.
    
    Extractive model trả về câu gốc → ít cần postprocessing hơn.
    Chỉ cần clean whitespace và đảm bảo kết thúc bằng dấu chấm.
    """
    
    def postprocess(self, summary: str, **kwargs) -> dict:
        # Clean whitespace cơ bản
        cleaned = clean_output(summary)
        
        return {
            "summary": cleaned,
            "original_length": len(summary),
            "processed_length": len(cleaned),
            "model_type": "extractive"
        }


# Factory function
def get_postprocessor(model_type: str) -> BasePostprocessor:
    """Lấy postprocessor phù hợp với loại model"""
    postprocessors = {
        "vit5_fin": ViT5FinPostprocessor(),
        "qwen": QwenPostprocessor(),
        "phobert_finance": PhoBertFinancePostprocessor(),
    }
    
    if model_type not in postprocessors:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return postprocessors[model_type]
