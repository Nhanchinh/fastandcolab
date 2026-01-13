"""
Schemas cho Evaluation API
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class EvaluateSingleRequest(BaseModel):
    """Request đánh giá một cặp prediction-reference"""
    prediction: str = Field(..., min_length=1, description="Văn bản tóm tắt (generated)")
    reference: str = Field(..., min_length=1, description="Văn bản tham khảo (ground truth)")
    calculate_bert: bool = Field(default=True, description="Có tính BERTScore không (chậm nhưng chính xác)")


class EvaluateSingleResponse(BaseModel):
    """Response từ API đánh giá đơn"""
    rouge1: float = Field(..., description="ROUGE-1 F1 score (0-1)")
    rouge2: float = Field(..., description="ROUGE-2 F1 score (0-1)")
    rougeL: float = Field(..., description="ROUGE-L F1 score (0-1)")
    bleu: float = Field(..., description="BLEU score (0-1)")
    bert_score: float = Field(..., description="BERTScore F1 (0-1), 0 nếu không tính")
    processing_time_ms: int = Field(..., description="Thời gian xử lý (ms)")


class EvaluateBatchRequest(BaseModel):
    """Request đánh giá batch predictions"""
    predictions: List[str] = Field(..., min_length=1, description="Danh sách văn bản tóm tắt")
    references: List[str] = Field(..., min_length=1, description="Danh sách văn bản tham khảo")
    calculate_bert: bool = Field(default=True, description="Có tính BERTScore không")
    batch_size: int = Field(default=16, ge=1, le=64, description="Batch size cho BERTScore")


class EvaluateBatchResponse(BaseModel):
    """Response từ API đánh giá batch"""
    avg_rouge1: float = Field(..., description="Average ROUGE-1 F1 score")
    avg_rouge2: float = Field(..., description="Average ROUGE-2 F1 score")
    avg_rougeL: float = Field(..., description="Average ROUGE-L F1 score")
    avg_bleu: float = Field(..., description="Average BLEU score")
    avg_bert_score: float = Field(..., description="Average BERTScore F1")
    avg_processing_time_ms: int = Field(..., description="Average processing time per sample (ms)")
    total_samples: int = Field(..., description="Số lượng samples đã đánh giá")


class SummarizeAndEvaluateRequest(BaseModel):
    """Request tóm tắt và đánh giá cùng lúc"""
    text: str = Field(..., min_length=10, description="Văn bản cần tóm tắt")
    reference: str = Field(..., min_length=1, description="Văn bản tham khảo để so sánh")
    model: str = Field(default="phobert_vit5", description="Model tóm tắt: phobert_vit5, vit5, qwen")
    max_length: int = Field(default=256, ge=50, le=512, description="Độ dài tối đa của tóm tắt")
    calculate_bert: bool = Field(default=True, description="Có tính BERTScore không")


class SummarizeAndEvaluateResponse(BaseModel):
    """Response từ API tóm tắt + đánh giá"""
    # Summarization results
    summary: str = Field(..., description="Văn bản tóm tắt được sinh ra")
    model_used: str = Field(..., description="Model đã sử dụng")
    inference_time_ms: float = Field(..., description="Thời gian inference (ms)")
    
    # Evaluation results
    rouge1: float
    rouge2: float
    rougeL: float
    bleu: float
    bert_score: float
    evaluation_time_ms: int = Field(..., description="Thời gian đánh giá (ms)")
    
    # Total
    total_time_ms: float = Field(..., description="Tổng thời gian xử lý (ms)")
