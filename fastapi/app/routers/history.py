"""
History Router
API endpoints cho quản lý lịch sử tóm tắt và feedback
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.schemas.history import (
    HistoryCreate,
    HistoryResponse,
    HistoryListResponse,
    FeedbackCreate,
    ExportDatasetResponse,
    RatingType,
    ModelType
)
from app.services.history_service import HistoryService, get_history_service


router = APIRouter(prefix="/history", tags=["history"])


@router.post("", response_model=HistoryResponse, status_code=status.HTTP_201_CREATED)
async def create_history(
    data: HistoryCreate,
    service: HistoryService = Depends(get_history_service)
) -> HistoryResponse:
    """
    Lưu lịch sử tóm tắt mới.
    
    - **input_text**: Văn bản gốc (tối thiểu 10 ký tự)
    - **summary**: Bản tóm tắt đã tạo
    - **model_used**: Model đã sử dụng (vit5, phobert_vit5, qwen)
    - **metrics**: Các thông số (words, compression, time)
    
    Returns:
        HistoryResponse với ID và thông tin entry đã lưu
    """
    try:
        return await service.save_history(data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi lưu history: {str(e)}"
        )


@router.get("", response_model=HistoryListResponse)
async def get_history_list(
    page: int = Query(default=1, ge=1, description="Số trang"),
    page_size: int = Query(default=20, ge=1, le=100, description="Số item mỗi trang"),
    model: Optional[ModelType] = Query(default=None, description="Filter theo model"),
    rating: Optional[RatingType] = Query(default=None, description="Filter theo rating"),
    has_feedback: Optional[bool] = Query(default=None, description="Chỉ lấy entries có/không có feedback"),
    from_date: Optional[datetime] = Query(default=None, description="Từ ngày (ISO format)"),
    to_date: Optional[datetime] = Query(default=None, description="Đến ngày (ISO format)"),
    service: HistoryService = Depends(get_history_service)
) -> HistoryListResponse:
    """
    Lấy danh sách lịch sử tóm tắt với filter và pagination.
    
    **Filters:**
    - model: Lọc theo model đã sử dụng
    - rating: Lọc theo đánh giá (good/bad/neutral)
    - has_feedback: Lọc entries có/không có feedback
    - from_date, to_date: Lọc theo khoảng thời gian
    
    Returns:
        HistoryListResponse với items, pagination info
    """
    return await service.get_history_list(
        page=page,
        page_size=page_size,
        model=model,
        rating=rating,
        has_feedback=has_feedback,
        from_date=from_date,
        to_date=to_date
    )


@router.get("/export/bad-summaries", response_model=ExportDatasetResponse)
async def export_bad_summaries(
    model: Optional[ModelType] = Query(default=None, description="Filter theo model"),
    limit: int = Query(default=100, ge=1, le=1000, description="Số lượng tối đa"),
    service: HistoryService = Depends(get_history_service)
) -> ExportDatasetResponse:
    """
    Export các bản tóm tắt được đánh giá 'bad' để tạo dataset training.
    
    **Use case:**
    - Tải về các bản tóm tắt tệ
    - Sử dụng corrected_summary để fine-tune model
    - Format phù hợp cho training pipeline
    
    Returns:
        ExportDatasetResponse với danh sách items
    """
    return await service.export_bad_summaries(model=model, limit=limit)


@router.get("/{history_id}", response_model=HistoryResponse)
async def get_history_detail(
    history_id: str,
    service: HistoryService = Depends(get_history_service)
) -> HistoryResponse:
    """
    Lấy chi tiết 1 history entry.
    
    Returns:
        HistoryResponse với đầy đủ thông tin + feedback nếu có
    """
    result = await service.get_history_by_id(history_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy history entry"
        )
    return result


@router.post("/{history_id}/feedback", response_model=HistoryResponse)
async def add_feedback(
    history_id: str,
    feedback: FeedbackCreate,
    service: HistoryService = Depends(get_history_service)
) -> HistoryResponse:
    """
    Thêm hoặc cập nhật feedback cho history entry.
    
    - **rating**: Đánh giá (good/bad/neutral) - bắt buộc
    - **comment**: Ghi chú (tùy chọn)
    - **corrected_summary**: Bản tóm tắt đã sửa - dùng cho training (tùy chọn)
    
    Returns:
        HistoryResponse đã cập nhật
    """
    result = await service.add_feedback(history_id, feedback)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy history entry"
        )
    return result


from app.schemas.history import BulkDeleteRequest, DeleteResponse


@router.delete("/{history_id}", response_model=DeleteResponse)
async def delete_history(
    history_id: str,
    service: HistoryService = Depends(get_history_service)
) -> DeleteResponse:
    """
    Xóa 1 history entry.
    """
    success = await service.delete_one(history_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy history entry"
        )
    return DeleteResponse(deleted_count=1, message="Đã xóa thành công")


@router.post("/bulk-delete", response_model=DeleteResponse)
async def bulk_delete(
    data: BulkDeleteRequest,
    service: HistoryService = Depends(get_history_service)
) -> DeleteResponse:
    """
    Xóa nhiều history entries theo danh sách IDs.
    """
    count = await service.delete_many(data.ids)
    return DeleteResponse(deleted_count=count, message=f"Đã xóa {count} mục")


@router.delete("/by-filter", response_model=DeleteResponse)
async def delete_by_filter(
    model: Optional[ModelType] = Query(default=None, description="Filter theo model"),
    rating: Optional[RatingType] = Query(default=None, description="Filter theo rating"),
    has_feedback: Optional[bool] = Query(default=None, description="Filter theo feedback"),
    service: HistoryService = Depends(get_history_service)
) -> DeleteResponse:
    """
    Xóa history entries theo filter (phải có ít nhất 1 filter).
    """
    if not any([model, rating, has_feedback is not None]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phải có ít nhất 1 filter (model, rating, hoặc has_feedback)"
        )
    
    count = await service.delete_by_filter(model=model, rating=rating, has_feedback=has_feedback)
    return DeleteResponse(deleted_count=count, message=f"Đã xóa {count} mục")


@router.delete("/all", response_model=DeleteResponse)
async def delete_all(
    confirm: bool = Query(..., description="Xác nhận xóa tất cả"),
    service: HistoryService = Depends(get_history_service)
) -> DeleteResponse:
    """
    Xóa tất cả history (nguy hiểm!). Phải truyền confirm=true.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phải xác nhận confirm=true để xóa tất cả"
        )
    
    count = await service.delete_all()
    return DeleteResponse(deleted_count=count, message=f"Đã xóa tất cả {count} mục")
