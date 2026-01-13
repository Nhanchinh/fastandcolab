"""
Text Postprocessing Utilities
Hậu xử lý kết quả tóm tắt từ các model
"""

import re
from typing import List
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


class ViT5Postprocessor(BasePostprocessor):
    """Postprocessor cho model ViT5"""
    
    def _fix_date_format(self, text: str) -> str:
        """
        Fix lỗi ngày tháng bị mất dấu /
        VD: 1852023 -> 18/5/2023, 206 -> 20/6, 245 -> 24/5
        """
        import re
        
        # Fix ngày ngắn (không có năm): 206 -> 20/6, 245 -> 24/5, 117 -> 11/7
        def fix_short_date(match):
            s = match.group(0)
            if len(s) == 3:  # dd/m format (206 -> 20/6)
                day, month = s[:2], s[2]
                if 1 <= int(day) <= 31 and 1 <= int(month) <= 9:
                    return f"{day}/{month}"
            elif len(s) == 4:  # dd/mm format (2406 -> 24/06)
                day, month = s[:2], s[2:]
                if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
                    return f"{day}/{month}"
            return s
        
        # Fix ngày đầy đủ (có năm): 1852023 -> 18/5/2023
        def fix_full_date(match):
            s = match.group(0)
            if len(s) == 7:  # dd/m/yyyy
                day, month, year = s[:2], s[2], s[3:]
                if 1 <= int(day) <= 31 and 1 <= int(month) <= 9:
                    return f"{day}/{month}/{year}"
                # Thử d/mm/yyyy
                day, month, year = s[0], s[1:3], s[3:]
                if 1 <= int(day) <= 9 and 1 <= int(month) <= 12:
                    return f"{day}/{month}/{year}"
            elif len(s) == 8:  # dd/mm/yyyy
                day, month, year = s[:2], s[2:4], s[4:]
                if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
                    return f"{day}/{month}/{year}"
            return s
        
        # Fix ngày đầy đủ trước (7-8 số)
        text = re.sub(r'\b(\d{7,8})\b', fix_full_date, text)
        
        # Fix ngày ngắn (3-4 số) - mở rộng context: ngày, là, trước, sau
        def replace_short(m):
            original = m.group(2)
            if len(original) == 3:
                day, month = original[:2], original[2]
                if 1 <= int(day) <= 31 and 1 <= int(month) <= 9:
                    return m.group(1) + f"{day}/{month}"
            elif len(original) == 4:
                day, month = original[:2], original[2:]
                if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
                    return m.group(1) + f"{day}/{month}"
            return m.group(0)
        
        text = re.sub(r'(ngày\s+|là\s+|trước\s+|sau\s+)(\d{3,4})\b', replace_short, text, flags=re.IGNORECASE)
        
        return text
    
    def postprocess(self, summary: str, **kwargs) -> dict:
        # Clean basic
        cleaned = clean_output(summary)
        
        # Fix date format (1852023 -> 18/5/2023)
        cleaned = self._fix_date_format(cleaned)
        
        # Remove repetition (ViT5 đôi khi lặp)
        cleaned = remove_repetition(cleaned)
        
        # Ensure complete sentences
        cleaned = ensure_complete_sentences(cleaned)
        
        return {
            "summary": cleaned,
            "original_length": len(summary),
            "processed_length": len(cleaned)
        }


class PhoBertViT5Postprocessor(BasePostprocessor):
    """Postprocessor cho model hybrid PhoBERT-ViT5"""
    
    def postprocess(
        self, 
        summary: str,
        selected_sentences: List[str] = None,
        **kwargs
    ) -> dict:
        # Clean basic
        cleaned = clean_output(summary)
        
        # Remove repetition
        cleaned = remove_repetition(cleaned)
        
        # Ensure complete sentences
        cleaned = ensure_complete_sentences(cleaned)
        
        metadata = {
            "summary": cleaned,
            "original_length": len(summary),
            "processed_length": len(cleaned)
        }
        
        if selected_sentences:
            metadata["num_selected_sentences"] = len(selected_sentences)
        
        return metadata


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


# Factory function
def get_postprocessor(model_type: str) -> BasePostprocessor:
    """Lấy postprocessor phù hợp với loại model"""
    postprocessors = {
        "vit5": ViT5Postprocessor(),
        "phobert_vit5": PhoBertViT5Postprocessor(),
        "phobert_vit5_paraphrase": PhoBertViT5Postprocessor(),  # Dùng chung postprocessor với phobert_vit5
        "qwen": QwenPostprocessor(),
    }
    
    if model_type not in postprocessors:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return postprocessors[model_type]
