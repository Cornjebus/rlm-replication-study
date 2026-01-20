"""
Hybrid labelers for OOLONG benchmark.

Cheap-first strategy:
1. Keyword matching (zero cost)
2. Embedding similarity (cheap)
3. RLM inference (expensive, fallback only)
"""

from .base import Labeler, LabelResult
from .keyword import KeywordLabeler
from .embedding import EmbeddingLabeler
from .rlm import RLMLabeler
from .gpt import GPTLabeler
from .hybrid import HybridLabeler, HybridLabelerConfig, HybridLabelerStats

__all__ = [
    'Labeler',
    'LabelResult',
    'KeywordLabeler',
    'EmbeddingLabeler',
    'RLMLabeler',
    'GPTLabeler',
    'HybridLabeler',
    'HybridLabelerConfig',
    'HybridLabelerStats',
]
