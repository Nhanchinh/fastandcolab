"""
Summarization Router
API endpoints cho chức năng tóm tắt văn bản
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional
from pydantic import BaseModel

from app.schemas.summarization import (
    SummarizeRequest,
    SummarizeResponse,
    ColabHealthResponse,
    AvailableModel,
    ModelType,
    CompareRequest,
    CompareResponse
)
from app.schemas.batch import BatchUploadResponse
from app.services.summarization_service import SummarizationService, get_summarization_service
from app.services.batch_service import BatchService, get_batch_service
from app.services.ai_judge_service import AIJudgeService, get_ai_judge_service


# AI Judge schemas
class AIJudgeSummary(BaseModel):
    model: str
    summary: str

class AIJudgeRequest(BaseModel):
    original_text: str
    summaries: List[AIJudgeSummary]

class AIJudgeRanking(BaseModel):
    model: str
    rank: int
    score: int
    reasoning: str

class AIJudgeResponse(BaseModel):
    winner: str
    rankings: List[AIJudgeRanking]
    detailed_analysis: str
    processing_time_ms: int


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


@router.post("/compare", response_model=CompareResponse)
async def compare_models(
    request: CompareRequest,
    service: SummarizationService = Depends(get_summarization_service)
) -> CompareResponse:
    """
    So sánh kết quả tóm tắt của nhiều models.
    
    - **text**: Văn bản cần tóm tắt
    - **models**: Danh sách models muốn so sánh (mặc định: vit5, phobert_vit5, qwen)
    - **max_length**: Độ dài tối đa
    
    Chạy tuần tự từng model để tiết kiệm RAM/GPU.
    
    Returns:
        CompareResponse với kết quả từ tất cả models
    """
    try:
        return await service.compare_models(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi so sánh models: {str(e)}"
        )


@router.post("/ai-judge", response_model=AIJudgeResponse)
async def ai_judge(
    request: AIJudgeRequest,
    service: AIJudgeService = Depends(get_ai_judge_service)
) -> AIJudgeResponse:
    """
    Sử dụng AI (Gemini) để so sánh và đánh giá các bản tóm tắt.
    
    - **original_text**: Văn bản gốc
    - **summaries**: Danh sách các bản tóm tắt cần so sánh [{model, summary}]
    
    AI sẽ phân tích theo 4 tiêu chí: Fluency, Coherence, Relevance, Consistency
    và xếp hạng các model từ tốt đến kém.
    
    Returns:
        AIJudgeResponse với model thắng, rankings và phân tích chi tiết
    """
    if not service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API chưa được cấu hình. Vui lòng set GEMINI_API_KEY environment variable."
        )
    
    if len(request.summaries) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cần ít nhất 2 bản tóm tắt để so sánh"
        )
    
    try:
        summaries_list = [{"model": s.model, "summary": s.summary} for s in request.summaries]
        result = await service.judge_summaries(request.original_text, summaries_list)
        return AIJudgeResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi AI Judge: {str(e)}"
        )


@router.get("/ai-models")
async def list_ai_models(
    service: AIJudgeService = Depends(get_ai_judge_service)
):
    """
    Liệt kê các models Gemini có sẵn cho API key.
    """
    models = service.list_available_models()
    return {"models": models}

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
