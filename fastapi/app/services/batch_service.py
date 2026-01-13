"""
Batch Processing Service
Xử lý file CSV/Excel và chạy summarization cho từng row
"""

import time
from typing import List, Optional, BinaryIO
from io import BytesIO

import pandas as pd

from app.schemas.summarization import ModelType, SummarizeRequest
from app.schemas.batch import BatchItemResult, BatchUploadResponse
from app.services.summarization_service import SummarizationService, get_summarization_service


class BatchService:
    """Service xử lý batch upload và evaluation"""
    
    def __init__(self, summarization_service: SummarizationService = None):
        self.summarization_service = summarization_service or get_summarization_service()
    
    def parse_file(
        self, 
        file_content: bytes, 
        filename: str,
        text_column: str = "text",
        reference_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parse file CSV/Excel thành DataFrame.
        
        Args:
            file_content: Nội dung file dạng bytes
            filename: Tên file để xác định loại
            text_column: Tên cột chứa văn bản
            reference_column: Tên cột chứa tóm tắt tham chiếu
        """
        file_buffer = BytesIO(file_content)
        
        # Xác định loại file và parse
        if filename.endswith('.csv'):
            df = pd.read_csv(file_buffer, encoding='utf-8')
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_buffer)
        else:
            raise ValueError(f"Unsupported file format: {filename}. Chỉ hỗ trợ CSV, XLSX, XLS.")
        
        # Validate columns
        if text_column not in df.columns:
            raise ValueError(f"Cột '{text_column}' không tồn tại. Các cột có sẵn: {list(df.columns)}")
        
        if reference_column and reference_column not in df.columns:
            raise ValueError(f"Cột reference '{reference_column}' không tồn tại.")
        
        return df
    
    async def process_batch(
        self,
        file_content: bytes,
        filename: str,
        model: ModelType,
        max_length: int = 256,
        text_column: str = "text",
        reference_column: Optional[str] = None
    ) -> BatchUploadResponse:
        """
        Xử lý batch file và trả về kết quả.
        
        Args:
            file_content: Nội dung file
            filename: Tên file
            model: Model sử dụng
            max_length: Độ dài tối đa tóm tắt
            text_column: Tên cột văn bản
            reference_column: Tên cột tham chiếu
        """
        start_time = time.time()
        
        # Parse file
        df = self.parse_file(file_content, filename, text_column, reference_column)
        
        results: List[BatchItemResult] = []
        successful = 0
        failed = 0
        
        # Process từng row
        for idx, row in df.iterrows():
            text = str(row[text_column])
            reference = str(row[reference_column]) if reference_column and pd.notna(row.get(reference_column)) else None
            
            try:
                # Tạo request
                request = SummarizeRequest(
                    text=text,
                    model=model,
                    max_length=max_length
                )
                
                # Gọi summarization service
                response = await self.summarization_service.summarize(request)
                
                results.append(BatchItemResult(
                    index=int(idx),
                    original_text=text,
                    summary=response.summary,
                    reference_summary=reference,
                    model_used=model,
                    inference_time_s=response.colab_inference_s,
                    success=True
                ))
                successful += 1
                
            except Exception as e:
                results.append(BatchItemResult(
                    index=int(idx),
                    original_text=text,
                    summary="",
                    reference_summary=reference,
                    model_used=model,
                    inference_time_s=0,
                    success=False,
                    error=str(e)
                ))
                failed += 1
        
        total_time = time.time() - start_time
        avg_time = total_time / len(results) if results else 0
        
        return BatchUploadResponse(
            total_items=len(results),
            successful_items=successful,
            failed_items=failed,
            model_used=model,
            total_time_s=round(total_time, 2),
            avg_time_per_item_s=round(avg_time, 2),
            results=results
        )


def get_batch_service() -> BatchService:
    """FastAPI dependency để inject BatchService"""
    return BatchService()
