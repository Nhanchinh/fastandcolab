"""
Batch Processing Schemas
Schemas cho upload CSV/Excel và batch evaluation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.schemas.summarization import ModelType


class BatchItemResult(BaseModel):
    """Kết quả tóm tắt cho 1 item trong batch"""
    index: int
    original_text: str
    summary: str
    reference_summary: Optional[str] = None  # Tóm tắt tham chiếu (nếu có)
    model_used: Optional[ModelType] = None
    inference_time_s: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    # Evaluation Metrics (Optional)
    # Evaluation Metrics (Optional)
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    bleu: Optional[float] = None
    bert_score: Optional[float] = None


class BatchUploadResponse(BaseModel):
    """Response cho batch upload API"""
    total_items: int
    successful_items: int
    failed_items: int
    model_used: Optional[ModelType] = None
    total_time_s: float
    avg_time_per_item_s: float
    results: List[BatchItemResult]


class BatchUploadRequest(BaseModel):
    """Request params cho batch upload (form data)"""
    model: ModelType = Field(default=ModelType.VIT5, description="Model sử dụng cho tất cả items")
    max_length: int = Field(default=256, ge=50, le=512)
    text_column: str = Field(default="text", description="Tên cột chứa văn bản cần tóm tắt")
    reference_column: Optional[str] = Field(default=None, description="Tên cột chứa tóm tắt tham chiếu (optional)")
