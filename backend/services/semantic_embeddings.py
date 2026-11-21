"""
Lucid Semantic Embedding Engine
Provides neural dense vector embeddings and semantic similarity matching
for claim-to-evidence cross-referencing and retrieval reranking.
"""
import os
import re
import logging
from typing import List, Tuple, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("lucid.embeddings")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [SemanticEmbedding] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Suppress HuggingFace symlinks and unauthenticated warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticEmbeddingEngine:
    """
    Computes semantic vector embeddings and cosine similarities
    using SentenceTransformers with automatic fallback to TF-IDF dense vectors.
    """

    def __init__(self):
        self._model = None
        self._model_load_attempted = False
        self._fallback_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            max_features=5000
        )

    def _get_model(self):
        """Lazy-load the SentenceTransformer model to optimize cold start."""
        if not self._model_load_attempted:
            self._model_load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}...")
                self._model = SentenceTransformer(MODEL_NAME)
                logger.info("SentenceTransformer model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize SentenceTransformer ({e}). Using TF-IDF semantic fallback.")
                self._model = None
        return self._model

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encodes string or list of strings into dense semantic embedding vectors."""
        if isinstance(texts, str):
            texts = [texts]

        cleaned = [t.strip() if t and t.strip() else "empty" for t in texts]
        model = self._get_model()

        if model is not None:
            try:
                embs = model.encode(cleaned, convert_to_numpy=True, show_progress_bar=False)
                return embs
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding error: {e}. Falling back to TF-IDF vectors.")

        # Fallback: TF-IDF vectorization
        try:
            vecs = self._fallback_vectorizer.fit_transform(cleaned).toarray()
            # Normalize
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms
        except Exception:
            return np.zeros((len(cleaned), 384))

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Computes semantic similarity between text1 and text2 in [0.0, 1.0].
        Captures paraphrased and reworded statements (e.g. 'Israeli forces shot Palestinian boy'
        vs 'IDF gunfire killed youth in West Bank').
        """
        if not text1 or not text2:
            return 0.0

        t1 = text1.strip()
        t2 = text2.strip()
        if not t1 or not t2:
            return 0.0

        model = self._get_model()
        if model is not None:
            try:
                from sentence_transformers import util
                emb1 = model.encode(t1, convert_to_tensor=True, show_progress_bar=False)
                emb2 = model.encode(t2, convert_to_tensor=True, show_progress_bar=False)
                score = util.cos_sim(emb1, emb2).item()
                # Clip to [0.0, 1.0]
                return max(0.0, min(1.0, float(score)))
            except Exception as e:
                logger.debug(f"Error computing neural similarity: {e}")

        # Lexical & TF-IDF hybrid fallback
        return self._lexical_semantic_similarity(t1, t2)

    def _lexical_semantic_similarity(self, text1: str, text2: str) -> float:
        """Hybrid n-gram and entity overlap fallback."""
        try:
            vecs = self._fallback_vectorizer.fit_transform([text1, text2])
            tfidf_sim = float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
        except Exception:
            tfidf_sim = 0.0

        # Word-level Jaccard overlap on significant stems
        w1 = set(re.findall(r'\b[a-z]{3,}\b', text1.lower()))
        w2 = set(re.findall(r'\b[a-z]{3,}\b', text2.lower()))
        stopwords = {
            'the', 'and', 'for', 'with', 'from', 'this', 'that', 'were', 'have', 'been',
            'which', 'their', 'about', 'after', 'before', 'reported', 'stated', 'said'
        }
        w1 -= stopwords
        w2 -= stopwords

        jaccard = len(w1 & w2) / max(1, len(w1 | w2))
        return max(0.0, min(1.0, 0.6 * tfidf_sim + 0.4 * jaccard))

    def rank_documents(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """
        Ranks candidate documents by semantic similarity to query.
        Returns list of (doc_index, similarity_score) sorted descending.
        """
        if not query or not documents:
            return []

        model = self._get_model()
        if model is not None:
            try:
                from sentence_transformers import util
                q_emb = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
                doc_embs = model.encode(documents, convert_to_tensor=True, show_progress_bar=False)
                scores = util.cos_sim(q_emb, doc_embs)[0].tolist()
                ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
                return [(idx, max(0.0, min(1.0, float(sc)))) for idx, sc in ranked]
            except Exception as e:
                logger.debug(f"Neural ranking error: {e}")

        # Fallback ranking
        ranked = []
        for idx, doc in enumerate(documents):
            sim = self._lexical_semantic_similarity(query, doc)
            ranked.append((idx, sim))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


_engine = SemanticEmbeddingEngine()


def get_semantic_engine() -> SemanticEmbeddingEngine:
    return _engine
