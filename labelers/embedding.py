"""
Embedding-based labeler using sentence transformers.

Uses local embeddings - no LLM API calls.
"""

import numpy as np
from typing import Optional
from .base import Labeler, LabelResult


# Descriptive prompts for each TREC category
# These help the embedding model understand category semantics
TREC_DESCRIPTIONS = {
    # Short codes
    'ABBR': 'What does this abbreviation or acronym stand for or mean',
    'DESC': 'Describe, explain, or define something, tell me why or how',
    'ENTY': 'Name a thing, object, entity, or type of something',
    'HUM': 'Who is a person, name someone, identify a human',
    'LOC': 'Where is a place, location, city, country, or region',
    'NUM': 'How many, what number, date, year, amount, or quantity',
    # OOLONG full names
    'abbreviation': 'What does this abbreviation or acronym stand for or mean',
    'description and abstract concept': 'Describe, explain, or define something, tell me why or how',
    'entity': 'Name a thing, object, entity, or type of something',
    'human being': 'Who is a person, name someone, identify a human',
    'location': 'Where is a place, location, city, country, or region',
    'numeric value': 'How many, what number, date, year, amount, or quantity'
}


class EmbeddingLabeler(Labeler):
    """
    Labels instances using embedding similarity.

    Uses sentence-transformers for local embeddings (no API calls).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", descriptions: dict = None):
        """
        Initialize with embedding model.

        Args:
            model_name: Sentence transformer model name
            descriptions: Optional custom label descriptions
        """
        self.model = None
        self.model_name = model_name
        self.descriptions = descriptions or TREC_DESCRIPTIONS
        self.label_embeddings_cache = {}

    def _ensure_model(self):
        """Lazy load model on first use."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )

    def _get_label_embeddings(self, vocabulary: list[str]) -> dict:
        """
        Get or compute label embeddings.

        Caches embeddings for efficiency.
        """
        cache_key = tuple(sorted(vocabulary))

        if cache_key not in self.label_embeddings_cache:
            self._ensure_model()
            embeddings = {}

            for label in vocabulary:
                # Use description if available, else just the label
                desc = self.descriptions.get(label, label)
                embeddings[label] = self.model.encode(desc)

            self.label_embeddings_cache[cache_key] = embeddings

        return self.label_embeddings_cache[cache_key]

    def label(self, instance: str, vocabulary: list[str]) -> LabelResult:
        """
        Label instance by embedding similarity.

        Args:
            instance: Question text to classify
            vocabulary: Available label options

        Returns:
            LabelResult with most similar label
        """
        self._ensure_model()
        label_embs = self._get_label_embeddings(vocabulary)

        # Encode instance
        instance_emb = self.model.encode(instance)

        # Compute cosine similarity to each label
        similarities = {}
        for label, emb in label_embs.items():
            # Cosine similarity
            sim = np.dot(instance_emb, emb) / (
                np.linalg.norm(instance_emb) * np.linalg.norm(emb)
            )
            similarities[label] = float(sim)

        best_label = max(similarities, key=similarities.get)
        confidence = similarities[best_label]

        # Normalize confidence to 0-1 range
        # Cosine similarity is already -1 to 1, shift to 0-1
        confidence = (confidence + 1) / 2

        return LabelResult(label=best_label, confidence=confidence, source="embedding")

    def label_batch(self, instances: list[str], vocabulary: list[str]) -> list[LabelResult]:
        """
        Batch label multiple instances efficiently.

        Uses batch encoding for speed.
        """
        self._ensure_model()
        label_embs = self._get_label_embeddings(vocabulary)

        # Batch encode all instances
        instance_embs = self.model.encode(instances)

        results = []
        for instance_emb in instance_embs:
            similarities = {}
            for label, emb in label_embs.items():
                sim = np.dot(instance_emb, emb) / (
                    np.linalg.norm(instance_emb) * np.linalg.norm(emb)
                )
                similarities[label] = float(sim)

            best_label = max(similarities, key=similarities.get)
            confidence = (similarities[best_label] + 1) / 2

            results.append(LabelResult(
                label=best_label,
                confidence=confidence,
                source="embedding"
            ))

        return results
