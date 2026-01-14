"""
Summarization Service
Orchestrate preprocessing -> Colab call -> postprocessing
"""

import time
from typing import Dict, Any

from app.schemas.summarization import (
    ModelType, 
    SummarizeRequest, 
    SummarizeResponse,
    CompareRequest,
    CompareResponse,
    ModelCompareResult
)
from app.services.colab_client import ColabClient, get_colab_client
from app.utils.preprocessing import get_preprocessor
from app.utils.postprocessing import get_postprocessor


class SummarizationService:
    """
    Service layer xử lý logic tóm tắt văn bản.
    - Tiền xử lý văn bản theo từng model
    - Gọi Colab server để inference
    - Hậu xử lý kết quả
    """
    
    def __init__(self, colab_client: ColabClient = None):
        self.colab_client = colab_client or get_colab_client()
    
    async def summarize(self, request: SummarizeRequest) -> SummarizeResponse:
        """
        Xử lý request tóm tắt văn bản.
        
        Pipeline:
        1. Preprocess text theo model type
        2. Gọi Colab API
        3. Postprocess kết quả
        """
        start_time = time.time()
        model_type = request.model.value
        
        # 1. Preprocessing
        preprocessor = get_preprocessor(model_type)
        preprocess_result = preprocessor.preprocess(
            text=request.text,
            max_length=request.max_length
        )
        
        # 2. Gọi Colab server
        colab_payload = {
            "text": preprocess_result.get("processed_text", request.text),
            "model": model_type,
            "max_length": request.max_length,
        }
        
        # Thêm sentences cho hybrid model
        if model_type == "phobert_vit5" and "sentences" in preprocess_result:
            colab_payload["preprocessed_sentences"] = preprocess_result["sentences"]
        
        colab_response = await self.colab_client.summarize(**colab_payload)
        
        # 3. Postprocessing
        postprocessor = get_postprocessor(model_type)
        postprocess_result = postprocessor.postprocess(
            summary=colab_response.get("summary", ""),
            **preprocess_result
        )
        
        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000
        colab_time_ms = colab_response.get("inference_time_ms", 0)
        
        return SummarizeResponse(
            original_text=request.text,
            preprocessed_text=preprocess_result.get("processed_text", request.text),
            summary=postprocess_result["summary"],
            model_used=request.model,
            colab_inference_ms=colab_time_ms,
            colab_inference_s=round(colab_time_ms / 1000, 2),
            total_processing_ms=total_time_ms,
            total_processing_s=round(total_time_ms / 1000, 2),
            metadata={
                "preprocess": {
                    k: v for k, v in preprocess_result.items() 
                    if k not in ["processed_text", "sentences"]
                },
                "postprocess": {
                    k: v for k, v in postprocess_result.items() 
                    if k != "summary"
                },
                "colab_model": colab_response.get("model_used", model_type)
            }
        )
    
    async def compare_models(self, request: CompareRequest) -> CompareResponse:
        """
        So sánh kết quả tóm tắt của nhiều models.
        Chạy tuần tự từng model để tiết kiệm RAM/GPU.
        """
        start_time = time.time()
        results = []
        
        # Preprocessing chung (dùng cho tất cả models)
        preprocessor = get_preprocessor("vit5")  # Dùng preprocessor chung
        preprocess_result = preprocessor.preprocess(
            text=request.text,
            max_length=request.max_length
        )
        
        # Chạy tuần tự từng model
        for model_type in request.models:
            model_start = time.time()
            
            try:
                # Tạo request cho từng model
                model_request = SummarizeRequest(
                    text=request.text,
                    model=model_type,
                    max_length=request.max_length
                )
                
                # Gọi summarize cho model này
                summary_response = await self.summarize(model_request)
                
                model_time_ms = (time.time() - model_start) * 1000
                
                results.append(ModelCompareResult(
                    model=model_type,
                    summary=summary_response.summary,
                    inference_time_ms=model_time_ms,
                    inference_time_s=round(model_time_ms / 1000, 2),
                    error=None
                ))
                
            except Exception as e:
                # Nếu model lỗi, vẫn tiếp tục với model khác
                model_time_ms = (time.time() - model_start) * 1000
                results.append(ModelCompareResult(
                    model=model_type,
                    summary="",
                    inference_time_ms=model_time_ms,
                    inference_time_s=round(model_time_ms / 1000, 2),
                    error=str(e)
                ))
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return CompareResponse(
            original_text=request.text,
            preprocessed_text=preprocess_result.get("processed_text", request.text),
            results=results,
            total_time_ms=total_time_ms,
            total_time_s=round(total_time_ms / 1000, 2),
            metadata={
                "models_count": len(request.models),
                "models": [m.value for m in request.models]
            }
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Kiểm tra kết nối đến Colab server"""
        return await self.colab_client.health_check()
    
    def get_available_models(self) -> list:
        """Lấy danh sách các model có sẵn"""
        return [
            {
                "id": ModelType.PHOBERT_VIT5.value,
                "name": "PhoBERT + ViT5 (Hybrid)",
                "description": "PhoBERT đánh giá độ quan trọng câu, ViT5 sinh tóm tắt từ top sentences"
            },
            {
                "id": ModelType.VIT5.value,
                "name": "ViT5",
                "description": "ViT5 thuần túy - sinh tóm tắt trực tiếp từ văn bản"
            },
            {
                "id": ModelType.QWEN.value,
                "name": "Qwen2.5-3B",
                "description": "Mô hình ngôn ngữ lớn Qwen 3B - hỗ trợ văn bản dài"
            }
        ]


# Dependency injection
def get_summarization_service() -> SummarizationService:
    """FastAPI dependency để inject SummarizationService"""
    return SummarizationService()
