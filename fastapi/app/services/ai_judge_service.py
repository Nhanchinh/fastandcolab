"""
AI Judge Service
Sử dụng Gemini API để đánh giá và so sánh các bản tóm tắt
Phân tích chi tiết từng model: ưu điểm, khuyết điểm, ý bị thiếu/sai
"""

import os
import json
import re
import time
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
    model_analyses: List[Dict[str, Any]]  # Phân tích chi tiết từng model
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
        Sử dụng Gemini để so sánh và đánh giá chi tiết các bản tóm tắt.
        Phân tích cụ thể: ý nào bị thiếu, sai, thừa; lý do model thắng/thua.
        """
        start_time = time.time()
        
        if not self.is_available():
            raise ValueError("Gemini API key chưa được cấu hình. Vui lòng set GEMINI_API_KEY environment variable.")
        
        # Build the prompt
        summaries_text = ""
        model_names_list = []
        for i, s in enumerate(summaries, 1):
            summaries_text += f"\n### Model {i}: {s['model']}\n{s['summary']}\n"
            model_names_list.append(s['model'])
        
        model_analyses_template = []
        for name in model_names_list:
            model_analyses_template.append(
                f'{{"model": "{name}", "strengths": ["<điểm mạnh 1>"], '
                f'"weaknesses": ["<điểm yếu 1>"], '
                f'"missing_points": ["<ý quan trọng bị thiếu>"], '
                f'"incorrect_points": ["<thông tin sai so với bản gốc>"], '
                f'"fluency_score": 80, "coherence_score": 80, '
                f'"relevance_score": 80, "consistency_score": 80}}'
            )
        
        prompt = f"""Bạn là một chuyên gia đánh giá chất lượng tóm tắt văn bản tiếng Việt.
Hãy phân tích CHI TIẾT từng bản tóm tắt, chỉ ra điểm mạnh, điểm yếu, ý bị thiếu, thông tin sai.

## VĂN BẢN GỐC:
{original_text}

## CÁC BẢN TÓM TẮT CẦN SO SÁNH:
{summaries_text}

## TIÊU CHÍ ĐÁNH GIÁ (mỗi tiêu chí 0-100):
1. **Fluency** (Trôi chảy): Văn phong tự nhiên, ngữ pháp đúng, câu văn mượt mà
2. **Coherence** (Mạch lạc): Các ý kết nối logic, dễ hiểu, có cấu trúc rõ ràng
3. **Relevance** (Liên quan): Giữ được các ý chính quan trọng, không thừa thông tin phụ
4. **Consistency** (Nhất quán): Không mâu thuẫn, không thêm thông tin không có trong bản gốc

## YÊU CẦU PHÂN TÍCH:
- Với MỖI bản tóm tắt, hãy chỉ ra CỤ THỂ:
  + Điểm mạnh (viết bằng tiếng Việt)
  + Điểm yếu (viết bằng tiếng Việt)
  + Các ý quan trọng từ bản gốc mà bản tóm tắt này BỎ QUÊN (nếu có)
  + Các thông tin SAI hoặc bóp méo so với bản gốc (nếu có)
- Giải thích RÕ RÀNG vì sao model thắng tốt hơn các model khác

## TRẢ VỀ JSON CHÍNH XÁC (không text thêm, chỉ JSON):
{{
    "winner": "<tên model thắng>",
    "rankings": [
        {{"model": "<tên model>", "rank": 1, "score": 85, "reasoning": "<lý do xếp hạng, 1-2 câu>"}},
        {{"model": "<tên model>", "rank": 2, "score": 70, "reasoning": "<lý do xếp hạng, 1-2 câu>"}}
    ],
    "model_analyses": [
        {{
            "model": "<tên model>",
            "strengths": ["<điểm mạnh 1>", "<điểm mạnh 2>"],
            "weaknesses": ["<điểm yếu 1>", "<điểm yếu 2>"],
            "missing_points": ["<ý quan trọng bị bỏ quên 1>"],
            "incorrect_points": ["<thông tin sai 1>"],
            "fluency_score": 85,
            "coherence_score": 80,
            "relevance_score": 75,
            "consistency_score": 90
        }}
    ],
    "detailed_analysis": "<so sánh tổng quan giữa các model, vì sao model thắng vượt trội, 3-4 câu>"
}}

Lưu ý:
- Score tổng từ 0-100, rank bắt đầu từ 1 (1 là tốt nhất)
- Nếu không có ý bị thiếu hoặc sai, ghi mảng rỗng []
- Viết TẤT CẢ bằng tiếng Việt
- model_analyses phải có đúng {len(summaries)} phần tử, mỗi model 1 phần tử"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
            
            result = json.loads(response_text)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Ensure model_analyses exists with fallback
            model_analyses = result.get("model_analyses", [])
            if not model_analyses:
                # Build default analyses from rankings if Gemini didn't include them
                model_analyses = []
                for r in result.get("rankings", []):
                    model_analyses.append({
                        "model": r.get("model", ""),
                        "strengths": [],
                        "weaknesses": [],
                        "missing_points": [],
                        "incorrect_points": [],
                        "fluency_score": r.get("score", 0),
                        "coherence_score": r.get("score", 0),
                        "relevance_score": r.get("score", 0),
                        "consistency_score": r.get("score", 0)
                    })
            
            return {
                "winner": result.get("winner", "unknown"),
                "rankings": result.get("rankings", []),
                "detailed_analysis": result.get("detailed_analysis", ""),
                "model_analyses": model_analyses,
                "processing_time_ms": processing_time
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Không thể parse response từ Gemini: {str(e)}")
        except Exception as e:
            raise ValueError(f"Lỗi khi gọi Gemini API: {str(e)}")
    
    async def generate_reference_summary(self, original_text: str) -> Dict[str, Any]:
        """
        Sử dụng Gemini để sinh bản tóm tắt gold (reference summary).
        Dùng làm ground truth để đánh giá chất lượng model.
        """
        start_time = time.time()
        
        if not self.is_available():
            raise ValueError("Gemini API key chưa được cấu hình.")
        
        prompt = f"""Bạn là một chuyên gia tóm tắt văn bản tiếng Việt. 
Hãy viết một bản tóm tắt CHẤT LƯỢNG CAO cho văn bản sau.

## VĂN BẢN GỐC:
{original_text}

## YÊU CẦU:
1. Viết thành MỘT ĐOẠN VĂN liền mạch, tự nhiên, dễ đọc
2. Giữ lại TẤT CẢ các ý chính quan trọng
3. KHÔNG bỏ sót thông tin cốt lõi
4. KHÔNG thêm thông tin không có trong bản gốc
5. Câu văn mạch lạc, ngữ pháp đúng
6. Viết bằng tiếng Việt
7. Độ dài khoảng 30-40% so với bản gốc

Chỉ trả về bản tóm tắt, không thêm giải thích hay ghi chú gì khác."""

        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            
            # Clean up markdown formatting if any
            summary = summary.strip('`').strip()
            if summary.startswith('```'):
                summary = summary.split('\n', 1)[-1]
            if summary.endswith('```'):
                summary = summary.rsplit('```', 1)[0]
            summary = summary.strip()
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "reference_summary": summary,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            raise ValueError(f"Lỗi khi sinh reference summary: {str(e)}")


# Singleton instance
_ai_judge_service: Optional[AIJudgeService] = None


def get_ai_judge_service() -> AIJudgeService:
    global _ai_judge_service
    if _ai_judge_service is None:
        _ai_judge_service = AIJudgeService()
    return _ai_judge_service
