"""
Summarization Router
API endpoints cho chức năng tóm tắt văn bản
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.schemas.summarization import (
    SummarizeRequest,
    SummarizeResponse,
    ColabHealthResponse,
    AvailableModel,
    ModelType
)
from app.services.summarization_service import SummarizationService, get_summarization_service


router = APIRouter(prefix="/summarization", tags=["summarization"])


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    request: SummarizeRequest,
    service: SummarizationService = Depends(get_summarization_service)
) -> SummarizeResponse:
    """
    Tóm tắt văn bản tiếng Việt.
    
    - **text**: Văn bản cần tóm tắt (tối thiểu 10 ký tự)
    - **model**: Loại model sử dụng (phobert_vit5, vit5, qwen)
    - **max_length**: Độ dài tối đa của bản tóm tắt (50-512)
    
    Returns:
        SummarizeResponse với bản tóm tắt và metadata
    """
    try:
        return await service.summarize(request)
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Không thể kết nối đến Colab server: {str(e)}"
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Colab server timeout: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý: {str(e)}"
        )


@router.get("/health", response_model=ColabHealthResponse)
async def check_health(
    service: SummarizationService = Depends(get_summarization_service)
) -> ColabHealthResponse:
    """
    Kiểm tra kết nối đến Colab GPU server.
    
    Returns:
        Status kết nối và thông tin GPU
    """
    result = await service.health_check()
    return ColabHealthResponse(**result)


@router.get("/models", response_model=List[AvailableModel])
async def get_models(
    service: SummarizationService = Depends(get_summarization_service)
) -> List[AvailableModel]:
    """
    Lấy danh sách các model tóm tắt có sẵn.
    
    Returns:
        Danh sách model với ID, tên và mô tả
    """
    models = service.get_available_models()
    return [AvailableModel(**m) for m in models]
