"""FAISS 向量检索引擎 — 农作物知识语义搜索"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import dotenv
dotenv.load_dotenv()

from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
AGRICULTURE_FAISS_PATH = os.getenv(
    "AGRICULTURE_FAISS_PATH",
    os.path.join(PROJECT_ROOT, "agriculture_faiss_index"),
)
FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    os.path.join(PROJECT_ROOT, "faiss_index"),
)


class FAISSAgricultureRAG:
    """基于 FAISS 向量检索的农业知识库，支持作物知识语义搜索"""

    def __init__(self, agriculture_index_path: str = None):
        self.agriculture_index_path = agriculture_index_path or AGRICULTURE_FAISS_PATH
        self._agri_store: Optional[FAISS] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._init_ok = bool(EMBEDDING_API_KEY)

    @property
    def is_available(self) -> bool:
        return self._init_ok and os.path.exists(self.agriculture_index_path)

    def _get_embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            kwargs = dict(model=EMBEDDING_MODEL)
            if EMBEDDING_BASE_URL:
                kwargs["base_url"] = EMBEDDING_BASE_URL
            if EMBEDDING_API_KEY:
                kwargs["api_key"] = EMBEDDING_API_KEY
            os.environ.setdefault("OPENAI_API_KEY", EMBEDDING_API_KEY or "")
            if EMBEDDING_BASE_URL:
                os.environ.setdefault("OPENAI_BASE_URL", EMBEDDING_BASE_URL)
            self._embeddings = OpenAIEmbeddings(**kwargs)
        return self._embeddings

    def _load_agriculture(self):
        if self._agri_store is not None:
            return
        if not os.path.exists(self.agriculture_index_path):
            return
        try:
            embeddings = self._get_embeddings()
            self._agri_store = FAISS.load_local(
                self.agriculture_index_path, embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS 作物知识索引加载成功")
        except Exception as e:
            logger.warning("FAISS 作物知识索引加载失败: %s", e)
            self._agri_store = None

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        向量语义搜索

        Returns:
            [{"content": str, "metadata": dict, "score": float}, ...]
        """
        if not self._init_ok:
            return []

        results = []

        # 农作物知识检索
        try:
            self._load_agriculture()
            if self._agri_store is not None:
                docs = self._agri_store.similarity_search_with_score(query, k=k)
                for doc, score in docs:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                    })
        except Exception as e:
            logger.warning("FAISS 农业知识检索出错: %s", e)

        # 按分数升序（FAISS 返回 L2 距离，越小越相似），去重
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x["score"]):
            fingerprint = r["content"][:80]
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(r)
        return unique[:k]
