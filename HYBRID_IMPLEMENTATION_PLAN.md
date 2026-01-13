# Hybrid RLM Implementation Plan

## Background

MIT feedback (Alex Zhang, RLM paper author) identified that our current evaluation uses `context_window_text_with_labels`, which exposes ground-truth labels. This makes the task pure aggregation (Counter operations) rather than the semantic inference + aggregation that OOLONG actually tests.

**The thesis we want to prove:**
> "Once semantic inference is done, aggregation should be deterministic, not recursive. RLM's recursive calls are only justified for the inference step, not for counting."

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      HYBRID PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PARSE (Deterministic)                                        │
│     context_window_text → [{date, user, instance}, ...]         │
│     No labels yet - just structured records                      │
│                                                                  │
│  2. LABEL (Cheap-First + RLM Fallback)                          │
│     ┌─────────────────────────────────────────────────┐         │
│     │  For each record:                                │         │
│     │    → Try keyword matcher (confidence?)           │         │
│     │    → Try embedding similarity (confidence?)      │         │
│     │    → If confidence < threshold:                  │         │
│     │        → Queue for RLM batch inference           │         │
│     └─────────────────────────────────────────────────┘         │
│     Output: [{date, user, instance, label, confidence}, ...]    │
│                                                                  │
│  3. AGGREGATE (Deterministic - Current Code)                    │
│     Counter operations on labeled records                        │
│     MOST_FREQ, LEAST_FREQ, RELATIVE_FREQ, etc.                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Infrastructure Changes

### 1.1 Create Label-Free Parser

**File:** `hybrid_benchmark.py`

```python
def parse_context_unlabeled(context_text: str, context_id: int, dataset: str) -> OolongContext:
    """Parse OOLONG context WITHOUT using labels."""
    records = []
    labels = []  # Available label vocabulary

    # Extract available labels from intro (we need vocabulary, not answers)
    label_match = re.search(r"one of (\d+) categories: (.+?)\.?\n", context_text)
    if label_match:
        label_str = label_match.group(2)
        labels = re.findall(r"'([^']+)'", label_str)

    # Parse records WITHOUT the label field
    # Format: Date: X || User: Y || Instance: Z
    record_pattern = r"Date:\s*([^|]+)\|\|\s*User:\s*(\d+)\s*\|\|\s*Instance:\s*([^|\n]+)"

    for match in re.finditer(record_pattern, context_text):
        records.append({
            'date': match.group(1).strip(),
            'user': match.group(2).strip(),
            'instance': match.group(3).strip(),
            'label': None,  # Must be inferred
            'confidence': 0.0
        })

    return OolongContext(
        context_id=context_id,
        dataset=dataset,
        records=records,
        labels=labels  # Vocabulary only
    )
```

### 1.2 Create Labeler Interface

**File:** `labelers/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LabelResult:
    label: str
    confidence: float
    source: str  # "keyword", "embedding", "rlm"

class Labeler(ABC):
    @abstractmethod
    def label(self, instance: str, vocabulary: list[str]) -> LabelResult:
        """Assign a label to an instance."""
        pass

    @abstractmethod
    def label_batch(self, instances: list[str], vocabulary: list[str]) -> list[LabelResult]:
        """Batch label multiple instances."""
        pass
```

## Phase 2: Cheap-First Labelers

### 2.1 Keyword Labeler (Zero LLM Calls)

**File:** `labelers/keyword.py`

For trec_coarse, the 6 categories are:
- ABBR (abbreviation)
- DESC (description)
- ENTY (entity)
- HUM (human)
- LOC (location)
- NUM (numeric)

```python
TREC_KEYWORDS = {
    'ABBR': ['abbreviation', 'acronym', 'stand for', 'short for'],
    'DESC': ['definition', 'describe', 'explanation', 'what is', 'why', 'how'],
    'ENTY': ['name', 'thing', 'object', 'type of', 'kind of'],
    'HUM': ['who', 'person', 'name of person', 'inventor', 'founder'],
    'LOC': ['where', 'city', 'country', 'place', 'location'],
    'NUM': ['how many', 'how much', 'number', 'date', 'year', 'percentage']
}

class KeywordLabeler(Labeler):
    def label(self, instance: str, vocabulary: list[str]) -> LabelResult:
        instance_lower = instance.lower()
        scores = {}

        for label, keywords in TREC_KEYWORDS.items():
            if label in vocabulary:
                score = sum(1 for kw in keywords if kw in instance_lower)
                scores[label] = score

        if not scores or max(scores.values()) == 0:
            return LabelResult(label="", confidence=0.0, source="keyword")

        best_label = max(scores, key=scores.get)
        # Confidence based on how many keywords matched
        confidence = min(scores[best_label] / 3.0, 1.0)

        return LabelResult(label=best_label, confidence=confidence, source="keyword")
```

### 2.2 Embedding Labeler (Zero LLM Calls, Uses Embeddings)

**File:** `labelers/embedding.py`

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingLabeler(Labeler):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.label_embeddings = {}

    def _get_label_embeddings(self, vocabulary: list[str]) -> dict:
        """Cache label embeddings."""
        cache_key = tuple(sorted(vocabulary))
        if cache_key not in self.label_embeddings:
            # Create descriptive prompts for each label
            label_descriptions = {
                'ABBR': 'What does this abbreviation stand for',
                'DESC': 'Describe or explain something',
                'ENTY': 'Name a thing, object, or entity',
                'HUM': 'Who is a person',
                'LOC': 'Where is a place or location',
                'NUM': 'How many or what number'
            }
            embeddings = {}
            for label in vocabulary:
                desc = label_descriptions.get(label, label)
                embeddings[label] = self.model.encode(desc)
            self.label_embeddings[cache_key] = embeddings
        return self.label_embeddings[cache_key]

    def label(self, instance: str, vocabulary: list[str]) -> LabelResult:
        label_embs = self._get_label_embeddings(vocabulary)
        instance_emb = self.model.encode(instance)

        similarities = {}
        for label, emb in label_embs.items():
            sim = np.dot(instance_emb, emb) / (np.linalg.norm(instance_emb) * np.linalg.norm(emb))
            similarities[label] = sim

        best_label = max(similarities, key=similarities.get)
        confidence = similarities[best_label]

        return LabelResult(label=best_label, confidence=confidence, source="embedding")
```

### 2.3 RLM Labeler (LLM Calls - Fallback Only)

**File:** `labelers/rlm.py`

```python
import anthropic

class RLMLabeler(Labeler):
    """
    RLM-style labeler - only called when cheap methods fail.
    Uses bounded recursion with explicit termination.
    """

    def __init__(self, max_retries: int = 1):
        self.client = anthropic.Anthropic()
        self.max_retries = max_retries
        self.call_count = 0

    def label(self, instance: str, vocabulary: list[str]) -> LabelResult:
        self.call_count += 1

        prompt = f"""Classify this question into exactly one category.

Categories: {', '.join(vocabulary)}

Question: {instance}

Respond with ONLY the category name, nothing else."""

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",  # Cheapest model
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )

        label = response.content[0].text.strip().upper()

        # Validate against vocabulary
        if label in vocabulary:
            return LabelResult(label=label, confidence=0.95, source="rlm")

        # Fallback: find closest match
        for v in vocabulary:
            if v in label or label in v:
                return LabelResult(label=v, confidence=0.7, source="rlm")

        return LabelResult(label=vocabulary[0], confidence=0.5, source="rlm")

    def label_batch(self, instances: list[str], vocabulary: list[str]) -> list[LabelResult]:
        """Batch multiple instances in one call for efficiency."""
        self.call_count += 1

        # Format batch prompt
        items = "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(instances))
        prompt = f"""Classify each question into exactly one category.

Categories: {', '.join(vocabulary)}

Questions:
{items}

Respond with ONLY the category names, one per line, in order."""

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=len(instances) * 20,
            messages=[{"role": "user", "content": prompt}]
        )

        labels = response.content[0].text.strip().split('\n')
        results = []

        for i, label in enumerate(labels):
            label = label.strip().upper()
            # Remove numbering if present
            label = re.sub(r'^\d+\.\s*', '', label)

            if label in vocabulary:
                results.append(LabelResult(label=label, confidence=0.95, source="rlm"))
            else:
                results.append(LabelResult(label=vocabulary[0], confidence=0.5, source="rlm"))

        # Pad if response was short
        while len(results) < len(instances):
            results.append(LabelResult(label=vocabulary[0], confidence=0.5, source="rlm"))

        return results[:len(instances)]
```

## Phase 3: Hybrid Orchestrator

**File:** `labelers/hybrid.py`

```python
from dataclasses import dataclass, field

@dataclass
class HybridLabelerConfig:
    keyword_threshold: float = 0.6      # Accept keyword if confidence >= this
    embedding_threshold: float = 0.7    # Accept embedding if confidence >= this
    batch_size: int = 20                # Batch RLM calls for efficiency
    max_rlm_calls_per_query: int = 5    # Cap RLM calls

@dataclass
class HybridLabelerStats:
    keyword_hits: int = 0
    embedding_hits: int = 0
    rlm_calls: int = 0
    rlm_records_labeled: int = 0
    total_records: int = 0

class HybridLabeler:
    def __init__(self, config: HybridLabelerConfig = None):
        self.config = config or HybridLabelerConfig()
        self.keyword = KeywordLabeler()
        self.embedding = EmbeddingLabeler()
        self.rlm = RLMLabeler()
        self.stats = HybridLabelerStats()

    def label_records(self, records: list[dict], vocabulary: list[str]) -> list[dict]:
        """
        Label all records using cheap-first strategy.

        Returns records with 'label' and 'confidence' filled in.
        """
        needs_rlm = []

        for record in records:
            self.stats.total_records += 1
            instance = record['instance']

            # Try keyword first (free)
            kw_result = self.keyword.label(instance, vocabulary)
            if kw_result.confidence >= self.config.keyword_threshold:
                record['label'] = kw_result.label
                record['confidence'] = kw_result.confidence
                record['label_source'] = 'keyword'
                self.stats.keyword_hits += 1
                continue

            # Try embedding (cheap)
            emb_result = self.embedding.label(instance, vocabulary)
            if emb_result.confidence >= self.config.embedding_threshold:
                record['label'] = emb_result.label
                record['confidence'] = emb_result.confidence
                record['label_source'] = 'embedding'
                self.stats.embedding_hits += 1
                continue

            # Queue for RLM
            needs_rlm.append(record)

        # Batch RLM calls for efficiency
        if needs_rlm:
            self._batch_rlm_label(needs_rlm, vocabulary)

        return records

    def _batch_rlm_label(self, records: list[dict], vocabulary: list[str]):
        """Batch RLM calls to minimize API usage."""
        batch_size = self.config.batch_size

        for i in range(0, len(records), batch_size):
            if self.stats.rlm_calls >= self.config.max_rlm_calls_per_query:
                # Hit cap - use best guess from embedding
                for record in records[i:]:
                    emb_result = self.embedding.label(record['instance'], vocabulary)
                    record['label'] = emb_result.label
                    record['confidence'] = emb_result.confidence
                    record['label_source'] = 'embedding_fallback'
                break

            batch = records[i:i+batch_size]
            instances = [r['instance'] for r in batch]

            results = self.rlm.label_batch(instances, vocabulary)
            self.stats.rlm_calls += 1
            self.stats.rlm_records_labeled += len(batch)

            for record, result in zip(batch, results):
                record['label'] = result.label
                record['confidence'] = result.confidence
                record['label_source'] = 'rlm'
```

## Phase 4: Updated Benchmark Runner

**File:** `hybrid_benchmark.py`

```python
def run_hybrid_benchmark(parquet_dir: str, config: HybridLabelerConfig = None):
    """
    Run hybrid benchmark using cheap-first + RLM fallback.
    """
    config = config or HybridLabelerConfig()

    # Load OOLONG data - trec_coarse only per MIT
    df = load_oolong_data(parquet_dir)
    df = df[df['dataset'] == 'trec_coarse']

    results = {
        'total': 0,
        'correct': 0,
        'zero_rlm_queries': 0,
        'total_rlm_calls': 0,
        'total_records_labeled': 0,
        'rlm_records_labeled': 0,
        'by_labeler': {'keyword': 0, 'embedding': 0, 'rlm': 0},
        'times': []
    }

    for idx, row in df.iterrows():
        start_time = time.time()

        # 1. PARSE (no labels)
        context = parse_context_unlabeled(
            row['context_window_text'],  # NOT with_labels!
            row['context_window_id'],
            row['dataset']
        )

        # 2. LABEL (hybrid)
        labeler = HybridLabeler(config)
        labeled_records = labeler.label_records(context.records, context.labels)
        context.records = labeled_records

        # Track stats
        results['total_rlm_calls'] += labeler.stats.rlm_calls
        results['total_records_labeled'] += labeler.stats.total_records
        results['rlm_records_labeled'] += labeler.stats.rlm_records_labeled
        results['by_labeler']['keyword'] += labeler.stats.keyword_hits
        results['by_labeler']['embedding'] += labeler.stats.embedding_hits
        results['by_labeler']['rlm'] += labeler.stats.rlm_records_labeled

        if labeler.stats.rlm_calls == 0:
            results['zero_rlm_queries'] += 1

        # 3. AGGREGATE (existing deterministic code)
        predicted = answer_oolong_query(
            context,
            row['task'],
            row['question'],
            row['answer_type']
        )

        elapsed = time.time() - start_time
        results['times'].append(elapsed)

        # Check answer
        is_correct = check_answer(predicted, row['answer'])
        results['total'] += 1
        if is_correct:
            results['correct'] += 1

    return results
```

## Phase 5: Evaluation Metrics

### 5.1 Primary Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Accuracy** | Exact-match on trec_coarse | ≥ RLM baseline (35-50% per paper) |
| **Zero-RLM Rate** | % queries with 0 RLM calls | As high as possible |
| **Avg RLM Calls** | Mean RLM calls per query | << RLM's 10-1000+ |
| **RLM Record Rate** | % records labeled by RLM | As low as possible |

### 5.2 Comparison Table (Template)

| Metric | Hybrid (Ours) | RLM (Zhang et al.) |
|--------|--------------|-------------------|
| Accuracy (exact-match) | TBD | 35-50% |
| Avg RLM calls/query | TBD | 10-1000+ |
| Zero-RLM queries | TBD% | 0% |
| Termination | Deterministic | Brittle |

## Phase 6: Implementation Order

### Step 1: Create directory structure
```bash
mkdir -p labelers
touch labelers/__init__.py
touch labelers/base.py
touch labelers/keyword.py
touch labelers/embedding.py
touch labelers/rlm.py
touch labelers/hybrid.py
touch hybrid_benchmark.py
```

### Step 2: Implement keyword labeler (no dependencies)
- Build TREC keyword mappings
- Test on sample records
- Measure baseline accuracy with keyword-only

### Step 3: Implement embedding labeler
- Install sentence-transformers
- Build label embeddings
- Test accuracy improvement over keyword-only

### Step 4: Implement RLM labeler
- Single-record and batch modes
- Call counting
- Bounded retries

### Step 5: Implement hybrid orchestrator
- Cheap-first cascade
- Confidence thresholds
- Stats tracking

### Step 6: Update benchmark runner
- Switch to `context_window_text`
- Integrate hybrid labeler
- Track all metrics

### Step 7: Run evaluation
- trec_coarse only (per MIT)
- Compare to RLM baseline
- Analyze where RLM calls are needed

### Step 8: Write up results
- Update REPLICATION_STUDY.md
- Address MIT feedback
- Publish comparison

## Expected Results

Based on the architecture, we expect:

1. **High zero-RLM rate** on "easy" instances where keyword/embedding works
2. **Bounded RLM calls** even on hard instances (batching + cap)
3. **Comparable or better accuracy** since aggregation is deterministic
4. **Proof of thesis**: Recursion only justified for inference, not counting

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Keyword labeler too simple | Tune keywords, fall through to embedding |
| Embedding threshold wrong | Grid search on validation set |
| RLM batch fails | Retry with smaller batches |
| Accuracy drops | Relax thresholds, accept more RLM calls |

## Success Criteria

1. **Accuracy**: At least match RLM's 35-50% on trec_coarse
2. **Efficiency**: ≤10 RLM calls average (vs RLM's 100+)
3. **Zero-RLM rate**: >50% queries need no RLM
4. **Deterministic aggregation**: 100% of aggregation done by Counter

## Timeline

Not providing time estimates per guidelines. The steps are ordered by dependency - complete each before moving to the next.
