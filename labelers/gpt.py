"""
GPT-based labeler for apples-to-apples comparison with RLM paper.

RLM paper used GPT-5 and achieved 56.50% on OOLONG.
"""

import os
import re
from typing import Optional
from .base import Labeler, LabelResult


class GPTLabeler(Labeler):
    """
    GPT-based labeler using OpenAI API.

    Used for direct comparison with RLM paper results.
    """

    def __init__(self, max_retries: int = 1, model: str = "gpt-5-chat-latest"):
        """
        Initialize GPT labeler.

        Args:
            max_retries: Max retry attempts per label
            model: OpenAI model to use
        """
        self.max_retries = max_retries
        self.model = model
        self.call_count = 0
        self.client = None

    def _ensure_client(self):
        """Lazy load OpenAI client."""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except ImportError:
                raise ImportError(
                    "openai required. Install with: pip install openai"
                )
            except Exception as e:
                if "OPENAI_API_KEY" in str(e):
                    raise ValueError(
                        "OPENAI_API_KEY environment variable required"
                    )
                raise

    def label(self, instance: str, vocabulary: list[str]) -> LabelResult:
        """
        Label a single instance using GPT.

        Args:
            instance: Question text to classify
            vocabulary: Available label options

        Returns:
            LabelResult with GPT's classification
        """
        self._ensure_client()
        self.call_count += 1

        prompt = f"""Classify this question into exactly one category.

Categories: {', '.join(vocabulary)}

Question: {instance}

Respond with ONLY the category name, nothing else."""

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=20,
                    messages=[{"role": "user", "content": prompt}]
                )

                label = response.choices[0].message.content.strip().lower()

                # Validate against vocabulary
                if label in vocabulary:
                    return LabelResult(label=label, confidence=0.95, source="gpt")

                # Try to find closest match
                for v in vocabulary:
                    if v in label or label in v:
                        return LabelResult(label=v, confidence=0.8, source="gpt")

            except Exception as e:
                if attempt == self.max_retries:
                    # Return low confidence on final failure
                    return LabelResult(
                        label=vocabulary[0] if vocabulary else "",
                        confidence=0.3,
                        source="gpt_error"
                    )

        # Fallback to first vocab item
        return LabelResult(label=vocabulary[0], confidence=0.5, source="gpt")

    def label_batch(self, instances: list[str], vocabulary: list[str]) -> list[LabelResult]:
        """
        Batch label multiple instances in one API call.

        More efficient than individual calls.

        Args:
            instances: List of questions to classify
            vocabulary: Available label options

        Returns:
            List of LabelResults
        """
        if not instances:
            return []

        self._ensure_client()
        self.call_count += 1

        # Format batch prompt
        items = "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(instances))
        prompt = f"""Classify each question into exactly one category.

Categories: {', '.join(vocabulary)}

Questions:
{items}

Respond with ONLY the category names, one per line, numbered to match.
Example format:
1. CATEGORY
2. CATEGORY
..."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=len(instances) * 30,
                messages=[{"role": "user", "content": prompt}]
            )

            raw_output = response.choices[0].message.content.strip()
            lines = raw_output.split('\n')

            results = []
            for i, line in enumerate(lines):
                if i >= len(instances):
                    break

                # Clean up the line
                label = line.strip().lower()
                # Remove numbering
                label = re.sub(r'^\d+[\.\)\-\s]+', '', label).strip()

                if label in vocabulary:
                    results.append(LabelResult(
                        label=label,
                        confidence=0.9,
                        source="gpt"
                    ))
                else:
                    # Find closest match
                    matched = False
                    for v in vocabulary:
                        if v in label or label in v:
                            results.append(LabelResult(
                                label=v,
                                confidence=0.7,
                                source="gpt"
                            ))
                            matched = True
                            break
                    if not matched:
                        results.append(LabelResult(
                            label=vocabulary[0],
                            confidence=0.5,
                            source="gpt"
                        ))

            # Pad if response was short
            while len(results) < len(instances):
                results.append(LabelResult(
                    label=vocabulary[0],
                    confidence=0.4,
                    source="gpt_incomplete"
                ))

            return results[:len(instances)]

        except Exception as e:
            # Return low confidence results on error
            return [
                LabelResult(
                    label=vocabulary[0] if vocabulary else "",
                    confidence=0.3,
                    source="gpt_error"
                )
                for _ in instances
            ]
