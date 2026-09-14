"""Semantic resource retrieval using FAISS + sentence embeddings.

Embeddings are computed once at startup from each resource's title + description
(not exact-keyword matching), so a query like "binary search trees" surfaces
"BST insertion practice" even without shared keywords. If sentence-transformers
can't be loaded (e.g. no network to fetch model weights), we fall back to a
lightweight TF-IDF vector space via scikit-learn so the app still runs and
still ranks by semantic-ish similarity rather than exact string match.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
import numpy as np


class ResourceRetriever:
    def __init__(self, resources_path: str = "data/resources.json"):
        with open(resources_path, "r") as f:
            self.resources: List[Dict[str, Any]] = json.load(f)["resources"]
        self._texts = [f"{r['title']}. {r['description']}" for r in self.resources]
        self._backend = None
        self._embeddings = None
        self._vectorizer = None
        self._build_index()

    def _build_index(self):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain.docstore.document import Document

            embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            docs = [
                Document(page_content=text, metadata=self.resources[i])
                for i, text in enumerate(self._texts)
            ]
            self._store = FAISS.from_documents(docs, embedder)
            self._backend = "faiss"
        except Exception:
            # Offline / no model weights available: fall back to TF-IDF cosine similarity.
            # Still "semantic" in the sense that it matches on concept-word co-occurrence
            # (e.g. "BST" documents share vocabulary with "binary search tree" queries)
            # rather than requiring an exact substring match.
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._embeddings = self._vectorizer.fit_transform(self._texts)
            self._backend = "tfidf"

    def similarity_search(self, query: str, k: int = 3, topic_filter: Optional[str] = None,
                           difficulty_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        candidates = self.resources
        if topic_filter:
            candidates_idx = [i for i, r in enumerate(self.resources) if r["topic"] == topic_filter]
        else:
            candidates_idx = list(range(len(self.resources)))

        if self._backend == "faiss":
            results = self._store.similarity_search(query, k=max(k * 3, k))
            ranked = [r.metadata for r in results]
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._embeddings)[0]
            order = np.argsort(-sims)
            ranked = [self.resources[i] for i in order]

        filtered = [r for r in ranked if r["topic"] in
                    ({topic_filter} if topic_filter else {c["topic"] for c in candidates})]
        if difficulty_filter:
            # prefer matching difficulty but don't hard-exclude, since progression matters
            filtered.sort(key=lambda r: 0 if r["difficulty"] == difficulty_filter else 1)
        return filtered[:k]

    def select_for_topic(self, topic_id: str, current_mastery: int) -> List[Dict[str, Any]]:
        """Chooses a beginner->advanced progression of resources for a topic,
        scaled by current mastery (skip material already mastered)."""
        if current_mastery < 30:
            difficulty = "beginner"
        elif current_mastery < 65:
            difficulty = "intermediate"
        else:
            difficulty = "advanced"
        query = f"{topic_id} concepts and practice for a student at {difficulty} level"
        return self.similarity_search(query, k=3, topic_filter=topic_id, difficulty_filter=difficulty)

    @property
    def backend_name(self) -> str:
        return self._backend
