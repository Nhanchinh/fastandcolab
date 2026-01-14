"""
History Schemas
Pydantic models cho History API endpoints
"""

from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# Types
RatingType = Literal["good", "bad", "neutral"]
ModelType = Literal["vit5", "phobert_vit5", "phobert_vit5_paraphrase", "qwen"]


# ============= Request Schemas =============

class HistoryCreate(BaseModel):
    """Schema để tạo history entry mới"""
    input_text: str = Field(..., min_length=10, description="Văn bản gốc")
    summary: str = Field(..., min_length=1, description="Bản tóm tắt")
    model_used: ModelType = Field(..., description="Model đã sử dụng")
    input_words: int = Field(default=0, ge=0)
    output_words: int = Field(default=0, ge=0)
    compression_ratio: float = Field(default=0.0, ge=0, le=100)
    processing_time_ms: int = Field(default=0, ge=0)
    colab_inference_ms: Optional[float] = Field(default=None, ge=0)


class FeedbackCreate(BaseModel):
    """Schema để thêm feedback cho history entry"""
    rating: RatingType = Field(..., description="Đánh giá: good/bad/neutral")
    comment: Optional[str] = Field(default=None, max_length=500)
    corrected_summary: Optional[str] = Field(
        default=None, 
        description="Bản tóm tắt đã sửa (dùng cho training)"
    )


class BulkDeleteRequest(BaseModel):
    """Schema để xóa nhiều entries"""
    ids: List[str] = Field(..., min_length=1, description="Danh sách IDs cần xóa")


class DeleteResponse(BaseModel):
    """Response khi xóa"""
    deleted_count: int
    message: str


# ============= Response Schemas =============

class MetricsResponse(BaseModel):
    """Response cho metrics"""
    input_words: int
    output_words: int
    compression_ratio: float
    processing_time_ms: int
    colab_inference_ms: Optional[float] = None


class FeedbackResponse(BaseModel):
    """Response cho feedback"""
    rating: RatingType
    comment: Optional[str] = None
    corrected_summary: Optional[str] = None
    feedback_at: datetime


class HistoryResponse(BaseModel):
    """Response cho 1 history entry"""
    id: str
    input_text: str
    summary: str
    model_used: ModelType
    created_at: datetime
    metrics: MetricsResponse
    feedback: Optional[FeedbackResponse] = None
    
    class Config:
        from_attributes = True


class HistoryListResponse(BaseModel):
    """Response cho danh sách history với pagination"""
    items: List[HistoryResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class HistoryFilters(BaseModel):
    """Filters cho GET /history"""
    model: Optional[ModelType] = None
    rating: Optional[RatingType] = None
    has_feedback: Optional[bool] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


# ============= Export Schemas =============

class ExportItem(BaseModel):
    """1 item trong export dataset"""
    input_text: str
    generated_summary: str
    corrected_summary: Optional[str] = None
    model_used: ModelType
    rating: RatingType
    comment: Optional[str] = None


class ExportDatasetResponse(BaseModel):
    """Response khi export bad summaries"""
    total_items: int
    items: List[ExportItem]
    exported_at: datetime
