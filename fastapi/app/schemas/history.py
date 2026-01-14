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


class HumanEvalScores(BaseModel):
    """Human evaluation scores (1-5 scale)"""
    fluency: Optional[int] = Field(default=None, ge=1, le=5, description="Độ trôi chảy")
    coherence: Optional[int] = Field(default=None, ge=1, le=5, description="Tính mạch lạc")
    relevance: Optional[int] = Field(default=None, ge=1, le=5, description="Tính liên quan")
    consistency: Optional[int] = Field(default=None, ge=1, le=5, description="Tính nhất quán")


class FeedbackCreate(BaseModel):
    """Schema để thêm feedback cho history entry"""
    rating: RatingType = Field(..., description="Đánh giá: good/bad/neutral")
    comment: Optional[str] = Field(default=None, max_length=500)
    corrected_summary: Optional[str] = Field(
        default=None, 
        description="Bản tóm tắt đã sửa (dùng cho training)"
    )
    human_eval: Optional[HumanEvalScores] = Field(
        default=None,
        description="Điểm đánh giá thủ công (Fluency, Coherence, Relevance, Consistency)"
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
    human_eval: Optional[HumanEvalScores] = None


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


class HumanEvalExportItem(BaseModel):
    """Item cho export human evaluation data"""
    summary: str
    model_used: ModelType
    created_at: datetime
    fluency: Optional[int] = None
    coherence: Optional[int] = None
    relevance: Optional[int] = None
    consistency: Optional[int] = None
    average_score: Optional[float] = None
    overall_rating: RatingType
    comment: Optional[str] = None


class HumanEvalExportResponse(BaseModel):
    """Response khi export human evaluation data"""
    total_items: int
    items: List[HumanEvalExportItem]
    exported_at: datetime


# ============= Analytics Schemas =============

class ModelStats(BaseModel):
    """Stats cho từng model"""
    model: str
    count: int
    avg_compression_ratio: float
    avg_processing_time_ms: float
    good_count: int
    bad_count: int
    neutral_count: int


class DailyCount(BaseModel):
    """Số lượng theo ngày"""
    date: str
    count: int


class AnalyticsResponse(BaseModel):
    """Response cho analytics dashboard"""
    total_summaries: int
    total_with_feedback: int
    feedback_rate: float
    rating_distribution: dict  # {"good": x, "bad": y, "neutral": z}
    model_distribution: dict  # {"vit5": x, "phobert_vit5": y, ...}
    model_stats: List[ModelStats]
    daily_counts: List[DailyCount]  # 30 ngày gần nhất
    avg_compression_ratio: float
    avg_processing_time_ms: float