"""
History Service
Business logic cho quản lý lịch sử tóm tắt và feedback
"""

from datetime import datetime
from typing import Dict, List, Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.connection import get_database
from app.schemas.history import (
    HistoryCreate,
    HistoryResponse,
    HistoryListResponse,
    FeedbackCreate,
    FeedbackResponse,
    MetricsResponse,
    ExportDatasetResponse,
    ExportItem,
    RatingType,
    ModelType
)


class HistoryService:
    """Service xử lý logic lịch sử tóm tắt"""
    
    COLLECTION_NAME = "summary_history"
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db[self.COLLECTION_NAME]
    
    async def save_history(self, data: HistoryCreate, user_id: Optional[str] = None) -> HistoryResponse:
        """Lưu lịch sử tóm tắt mới"""
        now = datetime.utcnow()
        
        doc = {
            "user_id": user_id,
            "input_text": data.input_text,
            "summary": data.summary,
            "model_used": data.model_used,
            "created_at": now,
            "metrics": {
                "input_words": data.input_words,
                "output_words": data.output_words,
                "compression_ratio": data.compression_ratio,
                "processing_time_ms": data.processing_time_ms,
                "colab_inference_ms": data.colab_inference_ms
            },
            "feedback": None
        }
        
        result = await self.collection.insert_one(doc)
        doc["_id"] = result.inserted_id
        
        return self._doc_to_response(doc)
    
    async def get_history_list(
        self,
        page: int = 1,
        page_size: int = 20,
        model: Optional[ModelType] = None,
        rating: Optional[RatingType] = None,
        has_feedback: Optional[bool] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        user_id: Optional[str] = None
    ) -> HistoryListResponse:
        """Lấy danh sách history với filter và pagination"""
        
        # Build filter query
        query: Dict = {}
        
        if user_id:
            query["user_id"] = user_id
        if model:
            query["model_used"] = model
        if rating:
            query["feedback.rating"] = rating
        if has_feedback is True:
            query["feedback"] = {"$ne": None}
        elif has_feedback is False:
            query["feedback"] = None
        if from_date:
            query["created_at"] = {"$gte": from_date}
        if to_date:
            if "created_at" in query:
                query["created_at"]["$lte"] = to_date
            else:
                query["created_at"] = {"$lte": to_date}
        
        # Count total
        total = await self.collection.count_documents(query)
        
        # Calculate pagination
        skip = (page - 1) * page_size
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        
        # Fetch documents
        cursor = self.collection.find(query).sort("created_at", -1).skip(skip).limit(page_size)
        docs = await cursor.to_list(length=page_size)
        
        items = [self._doc_to_response(doc) for doc in docs]
        
        return HistoryListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
    
    async def get_history_by_id(self, history_id: str) -> Optional[HistoryResponse]:
        """Lấy chi tiết 1 history entry"""
        try:
            doc = await self.collection.find_one({"_id": ObjectId(history_id)})
            if doc:
                return self._doc_to_response(doc)
            return None
        except Exception:
            return None
    
    async def add_feedback(self, history_id: str, feedback: FeedbackCreate) -> Optional[HistoryResponse]:
        """Thêm hoặc cập nhật feedback cho history entry"""
        try:
            now = datetime.utcnow()
            
            feedback_doc = {
                "rating": feedback.rating,
                "comment": feedback.comment,
                "corrected_summary": feedback.corrected_summary,
                "feedback_at": now
            }
            
            result = await self.collection.find_one_and_update(
                {"_id": ObjectId(history_id)},
                {"$set": {"feedback": feedback_doc}},
                return_document=True
            )
            
            if result:
                return self._doc_to_response(result)
            return None
        except Exception:
            return None
    
    async def export_bad_summaries(
        self,
        model: Optional[ModelType] = None,
        limit: int = 100
    ) -> ExportDatasetResponse:
        """Export các bản tóm tắt được đánh giá 'bad' để làm dataset training"""
        
        query: Dict = {"feedback.rating": "bad"}
        if model:
            query["model_used"] = model
        
        cursor = self.collection.find(query).sort("feedback.feedback_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        
        items: List[ExportItem] = []
        for doc in docs:
            feedback = doc.get("feedback", {})
            items.append(ExportItem(
                input_text=doc["input_text"],
                generated_summary=doc["summary"],
                corrected_summary=feedback.get("corrected_summary"),
                model_used=doc["model_used"],
                rating=feedback.get("rating", "bad"),
                comment=feedback.get("comment")
            ))
        
        return ExportDatasetResponse(
            total_items=len(items),
            items=items,
            exported_at=datetime.utcnow()
        )
    
    def _doc_to_response(self, doc: Dict) -> HistoryResponse:
        """Convert MongoDB document to response model"""
        metrics_data = doc.get("metrics", {})
        feedback_data = doc.get("feedback")
        
        feedback_response = None
        if feedback_data:
            feedback_response = FeedbackResponse(
                rating=feedback_data["rating"],
                comment=feedback_data.get("comment"),
                corrected_summary=feedback_data.get("corrected_summary"),
                feedback_at=feedback_data["feedback_at"]
            )
        
        return HistoryResponse(
            id=str(doc["_id"]),
            input_text=doc["input_text"],
            summary=doc["summary"],
            model_used=doc["model_used"],
            created_at=doc["created_at"],
            metrics=MetricsResponse(
                input_words=metrics_data.get("input_words", 0),
                output_words=metrics_data.get("output_words", 0),
                compression_ratio=metrics_data.get("compression_ratio", 0.0),
                processing_time_ms=metrics_data.get("processing_time_ms", 0),
                colab_inference_ms=metrics_data.get("colab_inference_ms")
            ),
            feedback=feedback_response
        )
    
    async def delete_one(self, history_id: str) -> bool:
        """Xóa 1 history entry"""
        try:
            result = await self.collection.delete_one({"_id": ObjectId(history_id)})
            return result.deleted_count > 0
        except Exception:
            return False
    
    async def delete_many(self, history_ids: List[str]) -> int:
        """Xóa nhiều history entries theo danh sách IDs"""
        try:
            object_ids = [ObjectId(hid) for hid in history_ids]
            result = await self.collection.delete_many({"_id": {"$in": object_ids}})
            return result.deleted_count
        except Exception:
            return 0
    
    async def delete_by_filter(
        self,
        model: Optional[ModelType] = None,
        rating: Optional[RatingType] = None,
        has_feedback: Optional[bool] = None
    ) -> int:
        """Xóa history entries theo filter"""
        query: Dict = {}
        
        if model:
            query["model_used"] = model
        if rating:
            query["feedback.rating"] = rating
        if has_feedback is True:
            query["feedback"] = {"$ne": None}
        elif has_feedback is False:
            query["feedback"] = None
        
        # Safety: require at least one filter
        if not query:
            return 0
        
        result = await self.collection.delete_many(query)
        return result.deleted_count
    
    async def delete_all(self) -> int:
        """Xóa tất cả history (dangerous!)"""
        result = await self.collection.delete_many({})
        return result.deleted_count


def get_history_service() -> HistoryService:
    """Dependency injection cho HistoryService"""
    db = get_database()
    return HistoryService(db)
