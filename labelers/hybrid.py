"""
Hybrid labeler orchestrating cheap-first strategy.

Order of operations:
1. Keyword matching (free)
2. Embedding similarity (cheap, local)
3. RLM inference (expensive, API calls)
"""

from dataclasses import dataclass, field
from typing import Optional
from .base import LabelResult
from .keyword import KeywordLabeler
from .embedding import EmbeddingLabeler
from .rlm import RLMLabeler


@dataclass
class HybridLabelerConfig:
    """Configuration for hybrid labeling."""
    # Confidence thresholds
    keyword_threshold: float = 0.5      # Accept keyword if confidence >= this
    embedding_threshold: float = 0.65   # Accept embedding if confidence >= this

    # RLM settings
    batch_size: int = 20                 # Batch RLM calls for efficiency
    max_rlm_calls_per_query: int = 10    # Cap RLM calls per query

    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    rlm_model: str = "claude-3-haiku-20240307"


@dataclass
class HybridLabelerStats:
    """Statistics from hybrid labeling run."""
    keyword_hits: int = 0
    embedding_hits: int = 0
    rlm_calls: int = 0
    rlm_records_labeled: int = 0
    total_records: int = 0

    @property
    def keyword_rate(self) -> float:
        """Fraction of records labeled by keyword."""
        return self.keyword_hits / self.total_records if self.total_records else 0

    @property
    def embedding_rate(self) -> float:
        """Fraction of records labeled by embedding."""
        return self.embedding_hits / self.total_records if self.total_records else 0

    @property
    def rlm_rate(self) -> float:
        """Fraction of records labeled by RLM."""
        return self.rlm_records_labeled / self.total_records if self.total_records else 0

    @property
    def zero_rlm(self) -> bool:
        """Whether no RLM calls were made."""
        return self.rlm_calls == 0


class HybridLabeler:
    """
    Orchestrates cheap-first labeling strategy.

    Tries labelers in order of cost:
    1. Keyword (free)
    2. Embedding (cheap)
    3. RLM (expensive)
    """

    def __init__(self, config: Optional[HybridLabelerConfig] = None):
        """
        Initialize hybrid labeler with config.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or HybridLabelerConfig()
        self.stats = HybridLabelerStats()

        # Initialize labelers lazily
        self._keyword = None
        self._embedding = None
        self._rlm = None

    @property
    def keyword(self) -> KeywordLabeler:
        """Lazy-load keyword labeler."""
        if self._keyword is None:
            self._keyword = KeywordLabeler()
        return self._keyword

    @property
    def embedding(self) -> EmbeddingLabeler:
        """Lazy-load embedding labeler."""
        if self._embedding is None:
            self._embedding = EmbeddingLabeler(model_name=self.config.embedding_model)
        return self._embedding

    @property
    def rlm(self) -> RLMLabeler:
        """Lazy-load RLM labeler."""
        if self._rlm is None:
            self._rlm = RLMLabeler(model=self.config.rlm_model)
        return self._rlm

    def label_records(self, records: list[dict], vocabulary: list[str]) -> list[dict]:
        """
        Label all records using cheap-first strategy.

        Modifies records in place, adding:
        - 'label': The assigned label
        - 'confidence': Confidence score (0-1)
        - 'label_source': Which labeler was used

        Args:
            records: List of record dicts with 'instance' key
            vocabulary: Available label options

        Returns:
            Same records list with labels added
        """
        needs_embedding = []
        needs_rlm = []

        # Phase 1: Try keyword matching (free)
        for record in records:
            self.stats.total_records += 1
            instance = record['instance']

            kw_result = self.keyword.label(instance, vocabulary)

            if kw_result.confidence >= self.config.keyword_threshold:
                record['label'] = kw_result.label
                record['confidence'] = kw_result.confidence
                record['label_source'] = 'keyword'
                self.stats.keyword_hits += 1
            else:
                needs_embedding.append(record)

        # Phase 2: Try embedding similarity (cheap, batch)
        if needs_embedding:
            instances = [r['instance'] for r in needs_embedding]
            emb_results = self.embedding.label_batch(instances, vocabulary)

            for record, result in zip(needs_embedding, emb_results):
                if result.confidence >= self.config.embedding_threshold:
                    record['label'] = result.label
                    record['confidence'] = result.confidence
                    record['label_source'] = 'embedding'
                    self.stats.embedding_hits += 1
                else:
                    needs_rlm.append(record)

        # Phase 3: RLM for remaining (expensive, batched)
        if needs_rlm:
            self._batch_rlm_label(needs_rlm, vocabulary)

        return records

    def _batch_rlm_label(self, records: list[dict], vocabulary: list[str]):
        """
        Batch RLM calls to minimize API usage.

        Respects max_rlm_calls_per_query cap.
        """
        batch_size = self.config.batch_size

        for i in range(0, len(records), batch_size):
            # Check if we've hit the cap
            if self.stats.rlm_calls >= self.config.max_rlm_calls_per_query:
                # Hit cap - use best guess from embedding
                for record in records[i:]:
                    emb_result = self.embedding.label(record['instance'], vocabulary)
                    record['label'] = emb_result.label
                    record['confidence'] = emb_result.confidence
                    record['label_source'] = 'embedding_fallback'
                break

            batch = records[i:i + batch_size]
            instances = [r['instance'] for r in batch]

            results = self.rlm.label_batch(instances, vocabulary)
            self.stats.rlm_calls += 1
            self.stats.rlm_records_labeled += len(batch)

            for record, result in zip(batch, results):
                record['label'] = result.label
                record['confidence'] = result.confidence
                record['label_source'] = result.source

    def reset_stats(self):
        """Reset statistics for new run."""
        self.stats = HybridLabelerStats()
        if self._rlm:
            self._rlm.call_count = 0
