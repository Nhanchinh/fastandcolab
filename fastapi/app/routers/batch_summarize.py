"""
Batch Summarize Router
API endpoint cho chức năng import Excel và tóm tắt hàng loạt với SSE progress
"""

import asyncio
import json
import time
import logging
from io import BytesIO
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from app.schemas.summarization import ModelType, SummarizeRequest
from app.services.summarization_service import SummarizationService, get_summarization_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch-summarize", tags=["batch-summarize"])


def parse_upload_file(file_content: bytes, filename: str, text_column: str) -> pd.DataFrame:
    """Parse file Excel/CSV thành DataFrame"""
    file_buffer = BytesIO(file_content)

    if filename.endswith('.csv'):
        df = pd.read_csv(file_buffer, encoding='utf-8')
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_buffer)
    else:
        raise ValueError(f"Không hỗ trợ định dạng file: {filename}. Chỉ hỗ trợ CSV, XLSX, XLS.")

    df.columns = df.columns.str.strip()

    if text_column not in df.columns:
        raise ValueError(f"Cột '{text_column}' không tồn tại. Các cột có: {list(df.columns)}")

    return df


@router.post("/preview")
async def preview_file(
    file: UploadFile = File(..., description="File CSV hoặc Excel"),
    text_column: str = Form(default="content", description="Tên cột chứa văn bản")
):
    """
    Preview nội dung file trước khi tóm tắt.
    Trả về danh sách cột, số dòng, và 5 dòng đầu tiên.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file CSV, XLSX, XLS")

    try:
        file_content = await file.read()
        file_buffer = BytesIO(file_content)

        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_buffer, encoding='utf-8')
        else:
            df = pd.read_excel(file_buffer)

        df.columns = df.columns.str.strip()

        # Lấy preview 5 dòng đầu
        preview_rows = []
        for idx, row in df.head(5).iterrows():
            preview_rows.append({col: str(val)[:200] for col, val in row.items()})

        return {
            "filename": file.filename,
            "columns": list(df.columns),
            "total_rows": len(df),
            "preview": preview_rows,
            "text_column_found": text_column in df.columns,
            "file_size_kb": round(len(file_content) / 1024, 1)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file: {str(e)}")


@router.post("/start")
async def start_batch_summarize(
    file: UploadFile = File(..., description="File CSV hoặc Excel"),
    model: str = Form(default="vit5_fin", description="Model tóm tắt"),
    max_length: int = Form(default=256, ge=50, le=512),
    text_column: str = Form(default="content", description="Tên cột chứa văn bản"),
    service: SummarizationService = Depends(get_summarization_service)
):
    """
    Bắt đầu batch summarization với SSE (Server-Sent Events) để stream progress.
    Client sẽ nhận từng kết quả realtime.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file CSV, XLSX, XLS")

    # Validate model
    try:
        model_type = ModelType(model)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Model không hợp lệ: {model}. Hỗ trợ: vit5_fin, qwen, phobert_finance"
        )

    try:
        file_content = await file.read()
        df = parse_upload_file(file_content, file.filename, text_column)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file: {str(e)}")

    total_rows = len(df)

    async def event_generator():
        """Generator SSE events cho từng item"""
        start_time = time.time()
        successful = 0
        failed = 0

        # Gửi event bắt đầu
        yield f"data: {json.dumps({'type': 'start', 'total': total_rows, 'model': model})}\n\n"

        for idx, row in df.iterrows():
            text = str(row[text_column]).strip()
            item_start = time.time()

            if not text or len(text) < 10 or text == 'nan':
                failed += 1
                event = {
                    "type": "item",
                    "index": int(idx),
                    "success": False,
                    "error": "Văn bản quá ngắn hoặc rỗng",
                    "original_text": text or "",
                    "summary": "",
                    "inference_time_s": 0,
                    "progress": round((int(idx) + 1) / total_rows * 100, 1),
                    "completed": int(idx) + 1,
                    "total": total_rows,
                    "successful": successful,
                    "failed": failed
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                continue

            try:
                request = SummarizeRequest(
                    text=text,
                    model=model_type,
                    max_length=max_length
                )

                response = await service.summarize(request)
                item_time = round(time.time() - item_start, 2)
                successful += 1

                event = {
                    "type": "item",
                    "index": int(idx),
                    "success": True,
                    "original_text": text,
                    "summary": response.summary,
                    "inference_time_s": item_time,
                    "progress": round((int(idx) + 1) / total_rows * 100, 1),
                    "completed": int(idx) + 1,
                    "total": total_rows,
                    "successful": successful,
                    "failed": failed
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            except Exception as e:
                item_time = round(time.time() - item_start, 2)
                failed += 1
                logger.error(f"Batch item {idx} error: {e}")

                event = {
                    "type": "item",
                    "index": int(idx),
                    "success": False,
                    "error": str(e)[:200],
                    "original_text": text,
                    "summary": "",
                    "inference_time_s": item_time,
                    "progress": round((int(idx) + 1) / total_rows * 100, 1),
                    "completed": int(idx) + 1,
                    "total": total_rows,
                    "successful": successful,
                    "failed": failed
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # Nhỏ delay để tránh overload
            await asyncio.sleep(0.1)

        # Gửi event hoàn thành
        total_time = round(time.time() - start_time, 2)
        done_event = {
            "type": "done",
            "total": total_rows,
            "successful": successful,
            "failed": failed,
            "total_time_s": total_time,
            "avg_time_s": round(total_time / total_rows, 2) if total_rows > 0 else 0
        }
        yield f"data: {json.dumps(done_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
