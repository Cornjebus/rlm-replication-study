"""
Hybrid labeler with GPT-5 as the RLM fallback.

Order of operations:
1. Keyword matching (free)
2. Embedding similarity (cheap, local)
3. GPT-5 inference (expensive, API calls)
"""

from dataclasses import dataclass
from typing import Optional
from .base import LabelResult
from .keyword import KeywordLabeler
from .embedding import EmbeddingLabeler
from .gpt import GPTLabeler


@dataclass
class HybridGPTLabelerConfig:
    """Configuration for hybrid GPT labeling."""
    # Confidence thresholds
    keyword_threshold: float = 0.5      # Accept keyword if confidence >= this
    embedding_threshold: float = 0.65   # Accept embedding if confidence >= this

    # GPT settings
    batch_size: int = 20                 # Batch GPT calls for efficiency
    max_gpt_calls_per_query: int = 10    # Cap GPT calls per query

    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    gpt_model: str = "gpt-5-chat-latest"


@dataclass
class HybridGPTLabelerStats:
    """Statistics from hybrid GPT labeling run."""
    keyword_hits: int = 0
    embedding_hits: int = 0
    gpt_calls: int = 0
    gpt_records_labeled: int = 0
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
    def gpt_rate(self) -> float:
        """Fraction of records labeled by GPT."""
        return self.gpt_records_labeled / self.total_records if self.total_records else 0


class HybridGPTLabeler:
    """
    Orchestrates cheap-first labeling strategy with GPT-5 fallback.

    Tries labelers in order of cost:
    1. Keyword (free)
    2. Embedding (cheap)
    3. GPT-5 (expensive)
    """

    def __init__(self, config: Optional[HybridGPTLabelerConfig] = None):
        """
        Initialize hybrid GPT labeler with config.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or HybridGPTLabelerConfig()
        self.stats = HybridGPTLabelerStats()

        # Initialize labelers lazily
        self._keyword = None
        self._embedding = None
        self._gpt = None

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
    def gpt(self) -> GPTLabeler:
        """Lazy-load GPT labeler."""
        if self._gpt is None:
            self._gpt = GPTLabeler(model=self.config.gpt_model)
        return self._gpt

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
        needs_gpt = []

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
                    needs_gpt.append(record)

        # Phase 3: GPT-5 for remaining (expensive, batched)
        if needs_gpt:
            self._batch_gpt_label(needs_gpt, vocabulary)

        return records

    def _batch_gpt_label(self, records: list[dict], vocabulary: list[str]):
        """
        Batch GPT calls to minimize API usage.

        Respects max_gpt_calls_per_query cap.
        """
        batch_size = self.config.batch_size

        for i in range(0, len(records), batch_size):
            # Check if we've hit the cap
            if self.stats.gpt_calls >= self.config.max_gpt_calls_per_query:
                # Hit cap - use best guess from embedding
                for record in records[i:]:
                    emb_result = self.embedding.label(record['instance'], vocabulary)
                    record['label'] = emb_result.label
                    record['confidence'] = emb_result.confidence
                    record['label_source'] = 'embedding_fallback'
                break

            batch = records[i:i + batch_size]
            instances = [r['instance'] for r in batch]

            results = self.gpt.label_batch(instances, vocabulary)
            self.stats.gpt_calls += 1
            self.stats.gpt_records_labeled += len(batch)

            for record, result in zip(batch, results):
                record['label'] = result.label
                record['confidence'] = result.confidence
                record['label_source'] = result.source

    def reset_stats(self):
        """Reset statistics for new run."""
        self.stats = HybridGPTLabelerStats()
        if self._gpt:
            self._gpt.call_count = 0
