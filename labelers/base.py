"""Base labeler interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LabelResult:
    """Result of a labeling operation."""
    label: str
    confidence: float
    source: str  # "keyword", "embedding", "rlm"


class Labeler(ABC):
    """Abstract base class for labelers."""

    @abstractmethod
    def label(self, instance: str, vocabulary: list[str]) -> LabelResult:
        """
        Assign a label to a single instance.

        Args:
            instance: The text to classify
            vocabulary: Available label options

        Returns:
            LabelResult with label, confidence, and source
        """
        pass

    def label_batch(self, instances: list[str], vocabulary: list[str]) -> list[LabelResult]:
        """
        Batch label multiple instances.

        Default implementation calls label() for each instance.
        Subclasses may override for efficiency.

        Args:
            instances: List of texts to classify
            vocabulary: Available label options

        Returns:
            List of LabelResults
        """
        return [self.label(inst, vocabulary) for inst in instances]
