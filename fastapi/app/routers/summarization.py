"""
Summarization Router
API endpoints cho chức năng tóm tắt văn bản
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional

from app.schemas.summarization import (
    SummarizeRequest,
    SummarizeResponse,
    ColabHealthResponse,
    AvailableModel,
    ModelType
)
from app.schemas.batch import BatchUploadResponse
from app.services.summarization_service import SummarizationService, get_summarization_service
from app.services.batch_service import BatchService, get_batch_service


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


@router.post("/batch-upload", response_model=BatchUploadResponse)
async def batch_upload(
    file: UploadFile = File(..., description="File CSV hoặc Excel chứa dataset"),
    model: str = Form(default="vit5", description="Model sử dụng: vit5, phobert_vit5, qwen"),
    max_length: int = Form(default=256, ge=50, le=512),
    text_column: str = Form(default="text", description="Tên cột chứa văn bản cần tóm tắt"),
    reference_column: Optional[str] = Form(default=None, description="Tên cột chứa tóm tắt tham chiếu (optional)"),
    batch_service: BatchService = Depends(get_batch_service)
) -> BatchUploadResponse:
    """
    Upload file CSV/Excel để đánh giá dataset lớn.
    
    **File format:**
    - CSV hoặc Excel (.xlsx, .xls)
    - Phải có cột chứa văn bản (mặc định: 'text')
    - Có thể có cột tóm tắt tham chiếu (optional)
    
    **Ví dụ CSV:**
    ```
    text,reference_summary
    "Văn bản cần tóm tắt 1","Tóm tắt tham chiếu 1"
    "Văn bản cần tóm tắt 2","Tóm tắt tham chiếu 2"
    ```
    
    Returns:
        BatchUploadResponse với kết quả cho từng item
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=400,
            detail="Chỉ hỗ trợ file CSV, XLSX, XLS"
        )
    
    # Validate model
    try:
        model_type = ModelType(model)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Supported: vit5, phobert_vit5, qwen"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process batch
        result = await batch_service.process_batch(
            file_content=file_content,
            filename=file.filename,
            model=model_type,
            max_length=max_length,
            text_column=text_column,
            reference_column=reference_column
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý batch: {str(e)}"
        )
