"""
Summary History Model
MongoDB document structure cho lưu trữ lịch sử tóm tắt và feedback
"""

from datetime import datetime
from typing import Literal, Optional, TypedDict


# Rating types
RatingType = Literal["good", "bad", "neutral"]

# Model types (matching existing schema)
ModelType = Literal["vit5", "phobert_vit5", "qwen"]


class SummaryMetrics(TypedDict, total=False):
    """Metrics của bản tóm tắt"""
    input_words: int
    output_words: int
    compression_ratio: float
    processing_time_ms: int
    colab_inference_ms: Optional[int]


class SummaryFeedback(TypedDict, total=False):
    """Feedback từ người dùng"""
    rating: RatingType
    comment: Optional[str]
    corrected_summary: Optional[str]  # Bản tóm tắt đã sửa (cho training)
    feedback_at: datetime


class SummaryHistoryDocument(TypedDict, total=False):
    """MongoDB document cho summary history"""
    _id: str
    user_id: Optional[str]           # User ID (nếu có auth)
    input_text: str                  # Văn bản gốc
    summary: str                     # Bản tóm tắt được tạo
    model_used: ModelType            # Model đã sử dụng
    created_at: datetime             # Thời gian tạo
    metrics: SummaryMetrics          # Các metrics
    feedback: Optional[SummaryFeedback]  # Feedback từ user
