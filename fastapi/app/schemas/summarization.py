from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ModelType(str, Enum):
    """Các loại model tóm tắt được hỗ trợ"""
    PHOBERT_VIT5 = "phobert_vit5"  # Hybrid: PhoBERT ranking + ViT5 generation
    PHOBERT_VIT5_PARAPHRASE = "phobert_vit5_paraphrase"  # Hybrid: PhoBERT + ViT5 Paraphrase
    VIT5 = "vit5"                   # Pure ViT5
    QWEN = "qwen"                   # Qwen2.5-7B


class SummarizeRequest(BaseModel):
    """Request body cho API tóm tắt"""
    text: str = Field(..., min_length=10, description="Văn bản tiếng Việt cần tóm tắt")
    model: ModelType = Field(default=ModelType.PHOBERT_VIT5, description="Model sử dụng")
    max_length: Optional[int] = Field(default=256, ge=50, le=512, description="Độ dài tối đa của bản tóm tắt")


class SummarizeResponse(BaseModel):
    """Response body từ API tóm tắt"""
    original_text: str
    preprocessed_text: str
    summary: str
    model_used: ModelType
    colab_inference_ms: float
    colab_inference_s: float  # Thời gian inference tính theo giây
    total_processing_ms: float
    total_processing_s: float  # Tổng thời gian tính theo giây
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ColabHealthResponse(BaseModel):
    """Response từ health check endpoint"""
    status: str
    colab_url: Optional[str] = None
    gpu_available: Optional[bool] = None
    error: Optional[str] = None


class AvailableModel(BaseModel):
    """Thông tin về model có sẵn"""
    id: ModelType
    name: str
    description: str
