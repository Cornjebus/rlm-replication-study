"""
Keyword-based labeler for TREC question classification.

Zero LLM calls - uses keyword matching only.
"""

from .base import Labeler, LabelResult


# TREC coarse categories with associated keywords
# Based on question classification patterns
TREC_KEYWORDS = {
    # Short codes
    'ABBR': [
        'abbreviation', 'acronym', 'stand for', 'short for',
        'what does', 'mean', 'stands for'
    ],
    'DESC': [
        'definition', 'describe', 'description', 'explanation', 'explain',
        'what is', 'what are', 'what was', 'what were',
        'why is', 'why are', 'why did', 'why do', 'why does',
        'how is', 'how are', 'how did', 'how do', 'how does',
        'what causes', 'what makes', 'what happened',
        'tell me about', 'meaning of'
    ],
    'ENTY': [
        'name a', 'name the', 'name of', 'what kind', 'what type',
        'which', 'what is the', 'what are the',
        'thing', 'object', 'item', 'entity',
        'called', 'known as'
    ],
    'HUM': [
        'who is', 'who are', 'who was', 'who were', 'who did',
        'person', 'people', 'inventor', 'founder', 'creator',
        'author', 'artist', 'president', 'leader', 'scientist',
        'actor', 'actress', 'singer', 'player', 'writer'
    ],
    'LOC': [
        'where is', 'where are', 'where was', 'where were', 'where did',
        'city', 'country', 'place', 'location', 'state', 'capital',
        'continent', 'region', 'address', 'origin', 'born',
        'located', 'situated'
    ],
    'NUM': [
        'how many', 'how much', 'how long', 'how old', 'how far',
        'how tall', 'how big', 'how fast', 'how often',
        'number', 'amount', 'quantity', 'count',
        'date', 'year', 'time', 'age', 'percentage', 'percent',
        'price', 'cost', 'rate', 'distance', 'size', 'weight',
        'population', 'score'
    ],
    # OOLONG full names (same keywords mapped)
    'abbreviation': [
        'abbreviation', 'acronym', 'stand for', 'short for',
        'what does', 'mean', 'stands for'
    ],
    'description and abstract concept': [
        'definition', 'describe', 'description', 'explanation', 'explain',
        'what is', 'what are', 'what was', 'what were',
        'why is', 'why are', 'why did', 'why do', 'why does',
        'how is', 'how are', 'how did', 'how do', 'how does',
        'what causes', 'what makes', 'what happened',
        'tell me about', 'meaning of'
    ],
    'entity': [
        'name a', 'name the', 'name of', 'what kind', 'what type',
        'which', 'what is the', 'what are the',
        'thing', 'object', 'item', 'entity',
        'called', 'known as'
    ],
    'human being': [
        'who is', 'who are', 'who was', 'who were', 'who did',
        'person', 'people', 'inventor', 'founder', 'creator',
        'author', 'artist', 'president', 'leader', 'scientist',
        'actor', 'actress', 'singer', 'player', 'writer'
    ],
    'location': [
        'where is', 'where are', 'where was', 'where were', 'where did',
        'city', 'country', 'place', 'location', 'state', 'capital',
        'continent', 'region', 'address', 'origin', 'born',
        'located', 'situated'
    ],
    'numeric value': [
        'how many', 'how much', 'how long', 'how old', 'how far',
        'how tall', 'how big', 'how fast', 'how often',
        'number', 'amount', 'quantity', 'count',
        'date', 'year', 'time', 'age', 'percentage', 'percent',
        'price', 'cost', 'rate', 'distance', 'size', 'weight',
        'population', 'score'
    ]
}


class KeywordLabeler(Labeler):
    """
    Labels instances based on keyword matching.

    Zero LLM calls - pure regex/string matching.
    """

    def __init__(self, keywords: dict = None):
        """
        Initialize with optional custom keywords.

        Args:
            keywords: Optional custom keyword dict, defaults to TREC_KEYWORDS
        """
        self.keywords = keywords or TREC_KEYWORDS

    def label(self, instance: str, vocabulary: list[str]) -> LabelResult:
        """
        Label instance by counting keyword matches.

        Args:
            instance: Question text to classify
            vocabulary: Available label options

        Returns:
            LabelResult with best matching label
        """
        instance_lower = instance.lower()
        scores = {}

        for label in vocabulary:
            if label not in self.keywords:
                scores[label] = 0
                continue

            keywords = self.keywords[label]
            # Count how many keywords appear in the instance
            score = sum(1 for kw in keywords if kw in instance_lower)
            scores[label] = score

        if not scores or max(scores.values()) == 0:
            return LabelResult(label="", confidence=0.0, source="keyword")

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        # Confidence based on:
        # - How many keywords matched (more = higher)
        # - How much better than second best (margin)
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

        # Base confidence from match count, boosted by margin
        confidence = min(best_score / 3.0, 0.8) + min(margin / 5.0, 0.2)
        confidence = min(confidence, 1.0)

        return LabelResult(label=best_label, confidence=confidence, source="keyword")
