"""
Evaluation Service - ROUGE, BLEU, BERTScore cho tiếng Việt
Ưu tiên gọi Colab GPU, fallback local nếu Colab không available
"""

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Service tính toán evaluation metrics cho văn bản tiếng Việt.
    
    **Architecture**:
    - Ưu tiên gọi Colab GPU server (BERTScore nhanh 10-50x)
    - Fallback tính local nếu Colab không available
    - Cache metrics instances (ROUGE, BLEU load một lần)
    
    **Vietnamese-Specific**:
    - ROUGE/BLEU: Cần word segmentation với pyvi
    - BERTScore: Dùng model multilingual, không cần tokenize thủ công
    """
    
    def __init__(self):
        """
        Khởi tạo service.
        Local metrics chỉ load khi cần (fallback).
        """
        # Local metrics (lazy loaded, chỉ dùng khi Colab unavailable)
        self._rouge = None
        self._bleu = None
        self._bert_metric = None
        self._bert_loaded = False
        
        # Colab client
        self._colab_client = None
        
        logger.info("EvaluationService initialized (Colab GPU mode)")
    
    def _get_colab_client(self):
        """Lazy init colab client"""
        if self._colab_client is None:
            from app.services.colab_client import get_colab_client
            self._colab_client = get_colab_client()
        return self._colab_client
    
    # ============ Colab GPU Methods (Primary) ============
    
    async def _evaluate_via_colab(
        self,
        predictions: List[str],
        references: List[str],
        calculate_bert: bool = True,
        batch_size: int = 32
    ) -> Optional[Dict[str, float]]:
        """
        Gọi Colab GPU để tính evaluation metrics.
        
        Returns:
            Dict metrics nếu thành công, None nếu Colab unavailable
        """
        try:
            colab = self._get_colab_client()
            result = await colab.evaluate(
                predictions=predictions,
                references=references,
                calculate_bert=calculate_bert,
                batch_size=batch_size
            )
            logger.info(f"Evaluation via Colab GPU: {result.get('processing_time_ms', 0):.0f}ms")
            return result
        except Exception as e:
            logger.warning(f"Colab evaluation failed, falling back to local: {e}")
            return None
    
    # ============ Local Fallback Methods ============
    
    def _load_rouge_bleu(self):
        """Load ROUGE và BLEU metrics locally (lightweight)"""
        if self._rouge is None:
            import evaluate
            logger.info("Loading ROUGE metric locally...")
            self._rouge = evaluate.load('rouge')
        
        if self._bleu is None:
            import evaluate
            logger.info("Loading BLEU metric locally...")
            self._bleu = evaluate.load('bleu')
    
    def _load_bertscore(self):
        """Lazy load BERTScore metric locally (heavy, ~700MB)"""
        if not self._bert_loaded:
            import evaluate
            logger.info("Loading BERTScore metric locally (this may take a while)...")
            self._bert_metric = evaluate.load("bertscore")
            self._bert_loaded = True
            logger.info("BERTScore loaded successfully (local)")
    
    def _preprocess_vietnamese(self, texts: List[str]) -> List[str]:
        """Tách từ tiếng Việt cho ROUGE/BLEU"""
        from pyvi import ViTokenizer
        return [ViTokenizer.tokenize(text) for text in texts]
    
    def _calculate_local(
        self,
        predictions: List[str],
        references: List[str],
        calculate_bert: bool = True,
        batch_size: int = 16
    ) -> Dict[str, float]:
        """Tính metrics cục bộ (fallback khi Colab unavailable)"""
        self._load_rouge_bleu()
        
        # ROUGE
        preds_tok = self._preprocess_vietnamese(predictions)
        refs_tok = self._preprocess_vietnamese(references)
        
        try:
            rouge_results = self._rouge.compute(
                predictions=preds_tok,
                references=refs_tok,
                use_stemmer=False
            )
            rouge1 = rouge_results['rouge1']
            rouge2 = rouge_results['rouge2']
            rougeL = rouge_results['rougeL']
        except Exception as e:
            logger.warning(f"Local ROUGE failed: {e}")
            rouge1 = rouge2 = rougeL = 0.0
        
        # BLEU
        try:
            refs_formatted = [[ref] for ref in refs_tok]
            bleu_results = self._bleu.compute(
                predictions=preds_tok,
                references=refs_formatted
            )
            bleu = bleu_results['bleu']
        except (ZeroDivisionError, Exception) as e:
            logger.warning(f"Local BLEU failed: {e}")
            bleu = 0.0
        
        # BERTScore (chậm trên CPU)
        bert_score = 0.0
        if calculate_bert:
            try:
                self._load_bertscore()
                results = self._bert_metric.compute(
                    predictions=predictions,
                    references=references,
                    lang="vi",
                    batch_size=batch_size,
                    verbose=False
                )
                bert_score = sum(results['f1']) / len(results['f1'])
            except Exception as e:
                logger.warning(f"Local BERTScore failed: {e}")
                bert_score = 0.0
        
        return {
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'bleu': bleu,
            'bert_score': bert_score
        }
    
    # ============ Public API ============
    
    async def evaluate_single(
        self,
        prediction: str,
        reference: str,
        calculate_bert: bool = True,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Đánh giá một cặp prediction-reference.
        Ưu tiên Colab GPU, fallback local.
        """
        start_time = time.time()
        
        # Handle empty inputs
        if not prediction or not reference or not prediction.strip() or not reference.strip():
            logger.warning(f"Empty prediction or reference detected.")
            return {
                'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
                'bleu': 0.0, 'bert_score': 0.0,
                'processing_time_ms': 0
            }
        
        preds = [prediction]
        refs = [reference]
        
        # Thử Colab GPU trước
        colab_result = await self._evaluate_via_colab(
            preds, refs, calculate_bert, batch_size=32
        )
        
        if colab_result is not None:
            processing_time = int((time.time() - start_time) * 1000)
            return {
                'rouge1': colab_result.get('rouge1', 0.0),
                'rouge2': colab_result.get('rouge2', 0.0),
                'rougeL': colab_result.get('rougeL', 0.0),
                'bleu': colab_result.get('bleu', 0.0),
                'bert_score': colab_result.get('bert_score', 0.0),
                'processing_time_ms': processing_time
            }
        
        # Fallback local
        logger.info("Using local evaluation (Colab unavailable)")
        result = self._calculate_local(preds, refs, calculate_bert, batch_size)
        processing_time = int((time.time() - start_time) * 1000)
        result['processing_time_ms'] = processing_time
        
        return result
    
    async def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        calculate_bert: bool = True,
        batch_size: int = 16,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, float]:
        """
        Đánh giá batch predictions.
        Ưu tiên Colab GPU, fallback local.
        """
        start_time = time.time()
        total_samples = len(predictions)
        
        logger.info(f"Starting batch evaluation for {total_samples} samples")
        
        # Thử Colab GPU trước
        colab_result = await self._evaluate_via_colab(
            predictions, references, calculate_bert, batch_size=32
        )
        
        if colab_result is not None:
            if progress_callback:
                await progress_callback(100)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                'avg_rouge1': colab_result.get('rouge1', 0.0),
                'avg_rouge2': colab_result.get('rouge2', 0.0),
                'avg_rougeL': colab_result.get('rougeL', 0.0),
                'avg_bleu': colab_result.get('bleu', 0.0),
                'avg_bert_score': colab_result.get('bert_score', 0.0),
                'avg_processing_time_ms': processing_time // total_samples,
                'total_samples': total_samples
            }
        
        # Fallback local
        logger.info("Using local batch evaluation (Colab unavailable)")
        
        if progress_callback:
            await progress_callback(10)
        
        result = self._calculate_local(predictions, references, calculate_bert, batch_size)
        
        if progress_callback:
            await progress_callback(100)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'avg_rouge1': result['rouge1'],
            'avg_rouge2': result['rouge2'],
            'avg_rougeL': result['rougeL'],
            'avg_bleu': result['bleu'],
            'avg_bert_score': result['bert_score'],
            'avg_processing_time_ms': processing_time // total_samples,
            'total_samples': total_samples
        }


# Singleton instance (lazy initialized)
_evaluation_service: Optional[EvaluationService] = None


def get_evaluation_service() -> EvaluationService:
    """
    Dependency injection cho FastAPI.
    Sử dụng singleton pattern để tránh load metrics nhiều lần.
    """
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service
