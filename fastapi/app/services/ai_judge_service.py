"""
AI Judge Service
Sử dụng Gemini API để đánh giá và so sánh các bản tóm tắt
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from pydantic import BaseModel


class AIJudgeRequest(BaseModel):
    original_text: str
    summaries: List[Dict[str, str]]  # [{"model": "vit5", "summary": "..."}, ...]


class AIJudgeResponse(BaseModel):
    winner: str  # model name
    rankings: List[Dict[str, Any]]  # [{"model": "...", "rank": 1, "score": 85, "reasoning": "..."}]
    detailed_analysis: str
    processing_time_ms: int


class AIJudgeService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # Use gemini-flash-lite-latest (free tier, high rate limit)
            self.model = genai.GenerativeModel('models/gemini-flash-lite-latest')
        else:
            self.model = None
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def list_available_models(self) -> List[str]:
        """List all available models for this API key"""
        if not self.api_key:
            return []
        try:
            models = genai.list_models()
            result = []
            for m in models:
                # Check if model supports generateContent
                methods = getattr(m, 'supported_generation_methods', [])
                if 'generateContent' in methods:
                    result.append(m.name)
            return result
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    async def judge_summaries(self, original_text: str, summaries: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Sử dụng Gemini để so sánh và đánh giá các bản tóm tắt.
        """
        import time
        start_time = time.time()
        
        if not self.is_available():
            raise ValueError("Gemini API key chưa được cấu hình. Vui lòng set GEMINI_API_KEY environment variable.")
        
        # Build the prompt
        summaries_text = ""
        for i, s in enumerate(summaries, 1):
            summaries_text += f"\n### Model {i}: {s['model']}\n{s['summary']}\n"
        
        prompt = f"""Bạn là một chuyên gia đánh giá chất lượng tóm tắt văn bản. 
Hãy so sánh các bản tóm tắt sau và xác định bản tóm tắt tốt nhất.

## VĂN BẢN GỐC:
{original_text}

## CÁC BẢN TÓM TẮT CẦN SO SÁNH:
{summaries_text}

## TIÊU CHÍ ĐÁNH GIÁ:
1. **Fluency** (Trôi chảy): Văn phong tự nhiên, ngữ pháp đúng
2. **Coherence** (Mạch lạc): Các ý kết nối logic, dễ hiểu
3. **Relevance** (Liên quan): Nội dung đúng trọng tâm, không thừa thãi
4. **Consistency** (Nhất quán): Không mâu thuẫn với văn bản gốc

## YÊU CẦU:
Trả về JSON với format CHÍNH XÁC như sau (không có text thêm, chỉ JSON):
{{
    "winner": "<tên model thắng>",
    "rankings": [
        {{"model": "<tên model>", "rank": 1, "score": 85, "reasoning": "<lý do ngắn gọn>"}},
        {{"model": "<tên model>", "rank": 2, "score": 75, "reasoning": "<lý do ngắn gọn>"}}
    ],
    "detailed_analysis": "<phân tích chi tiết tại sao model thắng tốt hơn, 2-3 câu>"
}}

Lưu ý: Score từ 0-100, rank bắt đầu từ 1 (1 là tốt nhất)."""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            # Sometimes Gemini wraps JSON in markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
            
            result = json.loads(response_text)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "winner": result.get("winner", "unknown"),
                "rankings": result.get("rankings", []),
                "detailed_analysis": result.get("detailed_analysis", ""),
                "processing_time_ms": processing_time
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Không thể parse response từ Gemini: {str(e)}")
        except Exception as e:
            raise ValueError(f"Lỗi khi gọi Gemini API: {str(e)}")


# Singleton instance
_ai_judge_service: Optional[AIJudgeService] = None


def get_ai_judge_service() -> AIJudgeService:
    global _ai_judge_service
    if _ai_judge_service is None:
        _ai_judge_service = AIJudgeService()
    return _ai_judge_service
