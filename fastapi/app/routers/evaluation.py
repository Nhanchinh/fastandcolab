"""
Evaluation Router - API endpoints cho đánh giá chất lượng tóm tắt
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, UploadFile, File, Form

from app.schemas.evaluation import (
    EvaluateSingleRequest,
    EvaluateSingleResponse,
    EvaluateBatchRequest,
    EvaluateBatchResponse,
    SummarizeAndEvaluateRequest,
    SummarizeAndEvaluateResponse
)
from app.schemas.summarization import ModelType, SummarizeRequest
from app.schemas.batch import BatchUploadResponse
from app.services.evaluation_service import EvaluationService, get_evaluation_service
from app.services.summarization_service import SummarizationService, get_summarization_service
from app.services.batch_service import BatchService, get_batch_service


router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.post(
    "/single",
    response_model=EvaluateSingleResponse,
    summary="Đánh giá một cặp prediction-reference",
    description="""
    Tính toán các metrics đánh giá cho một cặp văn bản:
    - **ROUGE-1, ROUGE-2, ROUGE-L**: Đo lường overlap n-gram
    - **BLEU**: Đo precision của n-grams
    - **BERTScore**: Đo semantic similarity (chậm nhưng chính xác)
    
    **Lưu ý**: BERTScore có thể mất 1-2 giây trên CPU.
    """,
)
async def evaluate_single(
    request: EvaluateSingleRequest,
    eval_service: EvaluationService = Depends(get_evaluation_service)
):
    """Đánh giá một cặp prediction-reference"""
    try:
        result = await eval_service.evaluate_single(
            prediction=request.prediction,
            reference=request.reference,
            calculate_bert=request.calculate_bert
        )
        return EvaluateSingleResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đánh giá: {str(e)}")


@router.post(
    "/batch",
    response_model=EvaluateBatchResponse,
    summary="Đánh giá batch predictions",
    description="""
    Đánh giá nhiều cặp prediction-reference cùng lúc.
    Tối ưu hóa cho batch processing với BERTScore.
    
    **Yêu cầu**: predictions và references phải có cùng độ dài.
    """,
)
async def evaluate_batch(
    request: EvaluateBatchRequest,
    eval_service: EvaluationService = Depends(get_evaluation_service)
):
    """Đánh giá batch predictions"""
    if len(request.predictions) != len(request.references):
        raise HTTPException(
            status_code=400,
            detail=f"predictions ({len(request.predictions)}) và references ({len(request.references)}) phải có cùng độ dài"
        )
    
    try:
        result = await eval_service.evaluate_batch(
            predictions=request.predictions,
            references=request.references,
            calculate_bert=request.calculate_bert,
            batch_size=request.batch_size
        )
        return EvaluateBatchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đánh giá batch: {str(e)}")


@router.post(
    "/summarize-and-evaluate",
    response_model=SummarizeAndEvaluateResponse,
    summary="Tóm tắt và đánh giá cùng lúc",
    description="""
    Kết hợp tóm tắt văn bản và đánh giá chất lượng trong một API call.
    
    Pipeline:
    1. Gửi văn bản đến Colab server để tóm tắt
    2. So sánh kết quả với reference text
    3. Trả về cả summary và evaluation metrics
    """,
)
async def summarize_and_evaluate(
    request: SummarizeAndEvaluateRequest,
    summarization_service: SummarizationService = Depends(get_summarization_service),
    eval_service: EvaluationService = Depends(get_evaluation_service)
):
    """Tóm tắt văn bản và đánh giá kết quả"""
    try:
        # 1. Tóm tắt văn bản
        summarize_request = SummarizeRequest(
            text=request.text,
            model=ModelType(request.model),
            max_length=request.max_length
        )
        summarize_result = await summarization_service.summarize(summarize_request)
        
        # 2. Đánh giá kết quả
        eval_result = await eval_service.evaluate_single(
            prediction=summarize_result.summary,
            reference=request.reference,
            calculate_bert=request.calculate_bert
        )
        
        # 3. Tính tổng thời gian
        total_time = summarize_result.total_processing_ms + eval_result['processing_time_ms']
        
        return SummarizeAndEvaluateResponse(
            summary=summarize_result.summary,
            model_used=summarize_result.model_used.value,
            inference_time_ms=summarize_result.colab_inference_ms,
            rouge1=eval_result['rouge1'],
            rouge2=eval_result['rouge2'],
            rougeL=eval_result['rougeL'],
            bleu=eval_result['bleu'],
            bert_score=eval_result['bert_score'],
            evaluation_time_ms=eval_result['processing_time_ms'],
            total_time_ms=total_time
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


@router.post(
    "/upload",
    response_model=BatchUploadResponse,
    summary="Upload file để đánh giá (Score Only)",
    description="""
    Upload file CSV/Excel chứa cột summary và reference để đánh giá chất lượng.
    
    **File format:**
    - CSV hoặc Excel (.xlsx, .xls)
    - Phải có cột 'summary' (văn bản tóm tắt)
    - Phải có cột 'reference' (tóm tắt tham chiếu)
    - Tên cột có thể tùy chỉnh
    """,
)
async def evaluate_file(
    file: UploadFile = File(..., description="File CSV hoặc Excel"),
    calculate_bert: bool = Form(default=False, description="Tính BERTScore (chậm)"),
    summary_column: str = Form(default="summary", description="Tên cột chứa văn bản tóm tắt"),
    reference_column: str = Form(default="reference", description="Tên cột chứa tóm tắt tham chiếu"),
    batch_service: BatchService = Depends(get_batch_service)
):
    """Đánh giá chất lượng tóm tắt từ file"""
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=400,
            detail="Chỉ hỗ trợ file CSV, XLSX, XLS"
        )
        
    try:
        # Read file content
        file_content = await file.read()
        
        # Process evaluation
        result = await batch_service.evaluate_from_file(
            file_content=file_content,
            filename=file.filename,
            calculate_bert=calculate_bert,
            summary_column=summary_column,
            reference_column=reference_column
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý file: {str(e)}"
        )
