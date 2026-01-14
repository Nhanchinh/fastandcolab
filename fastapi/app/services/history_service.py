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
    ModelType,
    AnalyticsResponse,
    ModelStats,
    DailyCount,
    HumanEvalScores,
    HumanEvalExportItem,
    HumanEvalExportResponse
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
            
            # Thêm human evaluation scores nếu có
            if feedback.human_eval:
                feedback_doc["human_eval"] = {
                    "fluency": feedback.human_eval.fluency,
                    "coherence": feedback.human_eval.coherence,
                    "relevance": feedback.human_eval.relevance,
                    "consistency": feedback.human_eval.consistency
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
    
    async def export_human_eval(
        self,
        model: Optional[ModelType] = None,
        limit: int = 500
    ) -> HumanEvalExportResponse:
        """Export các bản tóm tắt có human evaluation scores"""
        
        query: Dict = {"feedback.human_eval": {"$exists": True}}
        if model:
            query["model_used"] = model
        
        cursor = self.collection.find(query).sort("feedback.feedback_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        
        items: List[HumanEvalExportItem] = []
        for doc in docs:
            feedback = doc.get("feedback", {})
            human_eval = feedback.get("human_eval", {})
            
            # Calculate average score
            scores = [
                human_eval.get("fluency"),
                human_eval.get("coherence"),
                human_eval.get("relevance"),
                human_eval.get("consistency")
            ]
            valid_scores = [s for s in scores if s is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            
            items.append(HumanEvalExportItem(
                summary=doc["summary"],
                model_used=doc["model_used"],
                created_at=doc["created_at"],
                fluency=human_eval.get("fluency"),
                coherence=human_eval.get("coherence"),
                relevance=human_eval.get("relevance"),
                consistency=human_eval.get("consistency"),
                average_score=round(avg_score, 2) if avg_score else None,
                overall_rating=feedback.get("rating", "neutral"),
                comment=feedback.get("comment")
            ))
        
        return HumanEvalExportResponse(
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
            human_eval_data = feedback_data.get("human_eval")
            human_eval = None
            if human_eval_data:
                human_eval = HumanEvalScores(
                    fluency=human_eval_data.get("fluency"),
                    coherence=human_eval_data.get("coherence"),
                    relevance=human_eval_data.get("relevance"),
                    consistency=human_eval_data.get("consistency")
                )
            
            feedback_response = FeedbackResponse(
                rating=feedback_data["rating"],
                comment=feedback_data.get("comment"),
                corrected_summary=feedback_data.get("corrected_summary"),
                feedback_at=feedback_data["feedback_at"],
                human_eval=human_eval
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

    async def get_analytics(self) -> AnalyticsResponse:
        """Lấy analytics tổng quan cho dashboard"""
        from datetime import timedelta
        
        # Total summaries
        total_summaries = await self.collection.count_documents({})
        
        # Total with feedback
        total_with_feedback = await self.collection.count_documents({"feedback": {"$ne": None}})
        feedback_rate = (total_with_feedback / total_summaries * 100) if total_summaries > 0 else 0
        
        # Rating distribution
        rating_pipeline = [
            {"$match": {"feedback": {"$ne": None}}},
            {"$group": {"_id": "$feedback.rating", "count": {"$sum": 1}}}
        ]
        rating_cursor = self.collection.aggregate(rating_pipeline)
        rating_results = await rating_cursor.to_list(length=10)
        rating_distribution = {"good": 0, "bad": 0, "neutral": 0}
        for r in rating_results:
            if r["_id"] in rating_distribution:
                rating_distribution[r["_id"]] = r["count"]
        
        # Model distribution
        model_pipeline = [
            {"$group": {"_id": "$model_used", "count": {"$sum": 1}}}
        ]
        model_cursor = self.collection.aggregate(model_pipeline)
        model_results = await model_cursor.to_list(length=10)
        model_distribution = {m["_id"]: m["count"] for m in model_results if m["_id"]}
        
        # Model stats with ratings
        model_stats_pipeline = [
            {"$group": {
                "_id": "$model_used",
                "count": {"$sum": 1},
                "avg_compression_ratio": {"$avg": "$metrics.compression_ratio"},
                "avg_processing_time_ms": {"$avg": "$metrics.processing_time_ms"}
            }}
        ]
        model_stats_cursor = self.collection.aggregate(model_stats_pipeline)
        model_stats_results = await model_stats_cursor.to_list(length=10)
        
        model_stats: List[ModelStats] = []
        for ms in model_stats_results:
            if not ms["_id"]:
                continue
            # Count ratings for this model
            good_count = await self.collection.count_documents({
                "model_used": ms["_id"], 
                "feedback.rating": "good"
            })
            bad_count = await self.collection.count_documents({
                "model_used": ms["_id"], 
                "feedback.rating": "bad"
            })
            neutral_count = await self.collection.count_documents({
                "model_used": ms["_id"], 
                "feedback.rating": "neutral"
            })
            
            model_stats.append(ModelStats(
                model=ms["_id"],
                count=ms["count"],
                avg_compression_ratio=round(ms["avg_compression_ratio"] or 0, 2),
                avg_processing_time_ms=round(ms["avg_processing_time_ms"] or 0, 0),
                good_count=good_count,
                bad_count=bad_count,
                neutral_count=neutral_count
            ))
        
        # Daily counts (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        daily_pipeline = [
            {"$match": {"created_at": {"$gte": thirty_days_ago}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        daily_cursor = self.collection.aggregate(daily_pipeline)
        daily_results = await daily_cursor.to_list(length=31)
        daily_counts = [DailyCount(date=d["_id"], count=d["count"]) for d in daily_results]
        
        # Overall averages
        avg_pipeline = [
            {"$group": {
                "_id": None,
                "avg_compression_ratio": {"$avg": "$metrics.compression_ratio"},
                "avg_processing_time_ms": {"$avg": "$metrics.processing_time_ms"}
            }}
        ]
        avg_cursor = self.collection.aggregate(avg_pipeline)
        avg_results = await avg_cursor.to_list(length=1)
        
        avg_compression = 0
        avg_processing = 0
        if avg_results:
            avg_compression = round(avg_results[0].get("avg_compression_ratio") or 0, 2)
            avg_processing = round(avg_results[0].get("avg_processing_time_ms") or 0, 0)
        
        return AnalyticsResponse(
            total_summaries=total_summaries,
            total_with_feedback=total_with_feedback,
            feedback_rate=round(feedback_rate, 1),
            rating_distribution=rating_distribution,
            model_distribution=model_distribution,
            model_stats=model_stats,
            daily_counts=daily_counts,
            avg_compression_ratio=avg_compression,
            avg_processing_time_ms=avg_processing
        )


def get_history_service() -> HistoryService:
    """Dependency injection cho HistoryService"""
    db = get_database()
    return HistoryService(db)
