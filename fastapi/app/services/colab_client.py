"""
Colab Client - HTTP client kết nối đến Colab GPU Server
Tự động fetch URL từ GitHub Gist
"""

import os
from typing import Optional, Dict, Any

import httpx
from dotenv import load_dotenv

load_dotenv()


class ColabClient:
    """
    HTTP Client kết nối đến Colab GPU Server.
    URL được fetch động từ GitHub Gist.
    """
    
    def __init__(self):
        # URL đến raw file trên Gist chứa Colab URL
        self.gist_raw_url = os.getenv(
            "GIST_RAW_URL",
            "https://gist.githubusercontent.com/Nhanchinh/93a6c664f9af6258e9da0d20b8148b70/raw/colab_url.txt"
        )
        self.timeout = float(os.getenv("COLAB_TIMEOUT", "120"))
        self._cached_url: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy init HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Đóng HTTP client"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def get_colab_url(self, force_refresh: bool = False) -> str:
        """
        Lấy URL hiện tại của Colab server từ GitHub Gist.
        
        Args:
            force_refresh: Bỏ qua cache và lấy mới
            
        Returns:
            URL của Colab server (vd: https://xxxx.ngrok.io)
        """
        if self._cached_url and not force_refresh:
            return self._cached_url
        
        try:
            # Thêm timestamp để bypass cache
            url = f"{self.gist_raw_url}?t={int(__import__('time').time())}"
            resp = await self.client.get(url)
            resp.raise_for_status()
            self._cached_url = resp.text.strip()
            return self._cached_url
        except Exception as e:
            raise ConnectionError(f"Không thể lấy Colab URL từ Gist: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Kiểm tra kết nối đến Colab server.
        
        Returns:
            Dict với status và thông tin GPU
        """
        try:
            url = await self.get_colab_url(force_refresh=True)
            resp = await self.client.get(f"{url}/health")
            resp.raise_for_status()
            data = resp.json()
            return {
                "status": "connected",
                "colab_url": url,
                "gpu_available": data.get("gpu", False)
            }
        except Exception as e:
            return {
                "status": "disconnected",
                "colab_url": self._cached_url,
                "error": str(e)
            }
    
    async def summarize(
        self,
        text: str,
        model: str,
        max_length: int = 256,
        preprocessed_sentences: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Gọi API tóm tắt trên Colab server.
        
        Args:
            text: Văn bản cần tóm tắt (đã qua preprocessing)
            model: Loại model (phobert_vit5, vit5, qwen)
            max_length: Độ dài tối đa của tóm tắt
            preprocessed_sentences: Danh sách câu (cho hybrid model)
            
        Returns:
            Dict chứa summary và metadata từ Colab
        """
        url = await self.get_colab_url()
        
        payload = {
            "text": text,
            "model": model,
            "max_length": max_length,
            **kwargs
        }
        
        if preprocessed_sentences:
            payload["preprocessed_sentences"] = preprocessed_sentences
        
        try:
            resp = await self.client.post(f"{url}/summarize", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            raise TimeoutError(f"Colab server timeout sau {self.timeout}s")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Colab server lỗi: {e.response.text}")
        except Exception as e:
            raise ConnectionError(f"Không thể kết nối Colab server: {e}")


# Singleton instance
_colab_client: Optional[ColabClient] = None


def get_colab_client() -> ColabClient:
    """Lấy singleton instance của ColabClient"""
    global _colab_client
    if _colab_client is None:
        _colab_client = ColabClient()
    return _colab_client
