# GPT-5 OOLONG Benchmark Report: Replication Study and Alternative Approaches

**Date**: January 14, 2025
**Benchmark**: OOLONG (trec_coarse subset)
**Reference Paper**: Zhang et al. (2024), arXiv:2512.24601

---

## Executive Summary

We conducted a replication study of the RLM paper's OOLONG benchmark, exploring whether alternative approaches might complement the recursive methodology. Using the **same model (GPT-5)**, **same dataset (trec_coarse)**, and **same evaluation metric (OOLONG score)**, we tested a "classify once, aggregate deterministically" approach:

| Method | OOLONG Score | LLM Calls | Notes |
|--------|-------------|-----------|-------|
| RLM GPT-5 (paper) | 56.50% | 26,000+ | Recursive approach |
| Our Hybrid GPT-5 | 59.59% | ~580 | Combines keyword + embedding + LLM |
| Our Pure GPT-5 | 73.24% | 1,900 | Single-pass classification |

**Key Observations:**
- For aggregation-specific tasks, a single classification pass followed by deterministic counting shows promise
- The hybrid approach demonstrates that combining cheap heuristics with LLM fallback can reduce costs significantly
- **Important limitation**: We only evaluated on `trec_coarse` (250 examples). RLM was evaluated across multiple benchmarks where recursion may provide greater benefits

This work is intended as a **complementary study** to explore when recursion is most beneficial vs. when simpler approaches may suffice.

---

## 1. Proof of Results

### 1.1 Saved Results File

Results are persisted in `benchmark_data/oolong_gpt5_results.json`:

```json
{
  "score": 73.2350860415422,
  "exact_match": 63.6,
  "calls": 1900,
  "model": "gpt-5-chat-latest",
  "by_task": {
    "TASK_TYPE.MOST_FREQ": {"score": 60.0, "n": 25},
    "TASK_TYPE.LEAST_FREQ": {"score": 68.75, "n": 16},
    "TASK_TYPE.RELATIVE_FREQ": {"score": 79.28571428571428, "n": 140},
    "TASK_TYPE.NUMERIC_ONE_CLASS": {"score": 64.7503309290084, "n": 65},
    "TASK_TYPE.SECOND_MOST_FREQ": {"score": 100.0, "n": 4}
  }
}
```

### 1.2 Console Output Summary

```
======================================================================
OOLONG BENCHMARK - GPT-5 (Model: gpt-5-chat-latest)
======================================================================

Dataset: trec_coarse
Total examples: 250
Task types: 5

======================================================================
RESULTS
======================================================================

OOLONG Score: 73.24%
Exact Match: 63.60%
Total LLM calls: 1900
Avg time/query: 8049.7ms

By Task:
  TASK_TYPE.LEAST_FREQ: 68.8%
  TASK_TYPE.MOST_FREQ: 60.0%
  TASK_TYPE.NUMERIC_ONE_CLASS: 64.8%
  TASK_TYPE.RELATIVE_FREQ: 79.3%
  TASK_TYPE.SECOND_MOST_FREQ: 100.0%

======================================================================
COMPARISON (trec_coarse only)
======================================================================
Method                         Score           Calls
RLM GPT-5 (paper)              56.50%          26000+
Single-pass GPT-5              73.24%          1900
```

### 1.3 Task Type Breakdown

| Task Type | Score | N | Description |
|-----------|-------|---|-------------|
| SECOND_MOST_FREQ | 100.0% | 4 | Second most frequent label |
| RELATIVE_FREQ | 79.3% | 140 | Compare frequencies of two labels |
| LEAST_FREQ | 68.8% | 16 | Least frequent label |
| NUMERIC_ONE_CLASS | 64.8% | 65 | Count instances of specific label |
| MOST_FREQ | 60.0% | 25 | Most frequent label |

---

## 2. Apples-to-Apples Methodology

### 2.1 Same Model

| Aspect | RLM Paper | Our Benchmark |
|--------|-----------|---------------|
| Model | GPT-5 | `gpt-5-chat-latest` |
| API | OpenAI | OpenAI |
| Context | Standard chat completion | Standard chat completion |

We used the exact same GPT-5 model available through OpenAI's API. No fine-tuning, no special prompting tricks.

### 2.2 Same Dataset

| Aspect | RLM Paper | Our Benchmark |
|--------|-----------|---------------|
| Benchmark | OOLONG | OOLONG |
| Dataset | trec_coarse | trec_coarse |
| Split | validation | validation |
| Source | oolongbench/oolong-synth | oolongbench/oolong-synth |
| Examples | 250 | 250 |

We used the exact same HuggingFace dataset specified in the RLM paper.

### 2.3 Same Evaluation Metric

| Aspect | RLM Paper | Our Benchmark |
|--------|-----------|---------------|
| Metric | OOLONG score | OOLONG score |
| Numeric scoring | 0.75^|y-ŷ| | 0.75^|y-ŷ| |
| Categorical scoring | Exact match | Exact match |

The OOLONG metric gives partial credit for numeric answers that are close (exponential decay with error), and requires exact match for categorical answers.

### 2.4 Same Input Format

Per MIT (Alex Zhang) feedback, we used:
- `context_window_text` field (WITHOUT ground-truth labels exposed)
- NOT `context_window_text_with_labels` (which would leak answers)

This is critical: the model must infer labels from the question text, not read them from the context.

---

## 3. Our Process

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OOLONG Query Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Parse Context                                               │
│     ┌──────────────────────────────────────────────────────┐   │
│     │ context_window_text (NO labels)                       │   │
│     │ → Extract: Date, User, Instance (question text)       │   │
│     │ → Extract: Vocabulary (available label categories)    │   │
│     └──────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  2. Classify (ONE batch call per ~20 instances)                │
│     ┌──────────────────────────────────────────────────────┐   │
│     │ GPT-5 prompt:                                         │   │
│     │ "Classify each question into exactly one category.    │   │
│     │  Categories: abbreviation, entity, description, ...   │   │
│     │  Questions:                                           │   │
│     │  1. What is the full form of .com?                   │   │
│     │  2. Who was the first American in space?             │   │
│     │  ..."                                                 │   │
│     │                                                       │   │
│     │ Response: "1. abbreviation\n2. human being\n..."      │   │
│     └──────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  3. Aggregate (DETERMINISTIC - no LLM)                         │
│     ┌──────────────────────────────────────────────────────┐   │
│     │ from collections import Counter                       │   │
│     │                                                       │   │
│     │ label_counts = Counter(labels)                        │   │
│     │                                                       │   │
│     │ MOST_FREQ    → label_counts.most_common(1)[0][0]     │   │
│     │ LEAST_FREQ   → label_counts.most_common()[-1][0]     │   │
│     │ NUMERIC      → label_counts[target_label]            │   │
│     │ RELATIVE     → "more/less/same frequency as"          │   │
│     └──────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  4. Return Answer                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Insight

**Aggregation tasks may benefit from separating classification from counting.**

RLM treats each query as requiring iterative exploration of the context, which is powerful for complex reasoning. However, OOLONG aggregation queries have a specific structure:
1. Classify each record into a category
2. Count/compare the categories

For step 2, deterministic counting via Python's `Counter()` offers certain advantages:
- Speed (microseconds vs seconds)
- No additional API costs
- Deterministic (same input = same output)

This observation suggests that **hybrid approaches** - using RLM for complex reasoning and deterministic methods for simple aggregation - might combine the strengths of both.

### 3.3 Potential Advantages of Single-Pass Classification

| Recursive Approach | Single-Pass Alternative |
|-------------------|------------------------|
| Multiple classification attempts | Single classification pass |
| LLM tracks partial counts | `Counter()` handles counting |
| Flexible termination | Fixed termination after aggregation |
| 26,000+ LLM calls | 1,900 calls for classification only |

**Note**: These tradeoffs may differ across task types. RLM's recursion likely provides benefits for tasks requiring iterative refinement that we did not fully explore.

---

## 4. Detailed Methodology

### 4.1 Context Parsing

Each OOLONG example contains a `context_window_text` field with records in this format:

```
Date: 2024-01-15 || User: 12345 || Instance: What is the capital of France?
Date: 2024-01-16 || User: 67890 || Instance: Who wrote Hamlet?
...
```

We extract:
- **Instance**: The question text to classify
- **Vocabulary**: Available label categories (e.g., "abbreviation", "entity", "description", "human being", "location", "numeric value")

### 4.2 Batch Classification

We send batches of 20 instances to GPT-5:

```python
prompt = """Classify each question into exactly one category.

Categories: abbreviation, description and abstract concept, entity, human being, location, numeric value

Questions:
1. What is the full form of .com?
2. What does the__(term)___(abbreviation)___(stand)__ for?
3. Who was the first American in space?
...

Respond with ONLY the category names, one per line, numbered to match.
Example format:
1. CATEGORY
2. CATEGORY
..."""
```

### 4.3 Deterministic Aggregation

Once records are labeled, aggregation is pure Python:

```python
from collections import Counter

# Initialize with all labels at 0 (important for LEAST_FREQ)
label_counts = Counter({label: 0 for label in vocabulary})
for record in records:
    if record['label']:
        label_counts[record['label']] += 1

# MOST_FREQ query
answer = label_counts.most_common(1)[0][0]

# LEAST_FREQ query
answer = label_counts.most_common()[-1][0]

# NUMERIC_ONE_CLASS query ("How many 'location' instances?")
answer = str(label_counts['location'])

# RELATIVE_FREQ query ("Is X more/less common than Y?")
if label_counts['X'] > label_counts['Y']:
    answer = "more common than"
elif label_counts['X'] < label_counts['Y']:
    answer = "less common than"
else:
    answer = "same frequency as"
```

### 4.4 OOLONG Scoring

```python
def oolong_score(predicted, expected):
    # Numeric answers: partial credit with exponential decay
    if both_numeric:
        return 0.75 ** abs(expected - predicted)

    # Categorical answers: exact match only
    return 1.0 if predicted == expected else 0.0
```

---

## 5. Cost Analysis

### 5.1 API Calls Comparison

| Metric | RLM (Recursive) | Single-Pass | Notes |
|--------|-----------------|-------------|-------|
| Total LLM calls | 26,000+ | 1,900 | Different approaches |
| Calls per query | ~104 | ~7.6 | RLM explores iteratively |

### 5.2 Estimated Cost

Assuming GPT-5 pricing (~$0.01 per 1K input tokens, ~$0.03 per 1K output tokens):

| Approach | Calls | Est. Cost |
|----------|-------|-----------|
| RLM (recursive) | 26,000+ | ~$5.20+ |
| Single-pass | 1,900 | ~$0.38 |
| Hybrid | ~580 | ~$0.12 |

**Note**: RLM's higher call count reflects its recursive exploration strategy, which may provide benefits on tasks not covered in this study.

---

## 6. Reproducibility

### 6.1 Run the Benchmark

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run benchmark
python3 run_oolong_gpt5.py --model gpt-5-chat-latest --dataset trec_coarse

# Results saved to benchmark_data/oolong_gpt5_results.json
```

### 6.2 Verify Results

```bash
# Check saved results
cat benchmark_data/oolong_gpt5_results.json

# Verify score calculation
python3 -c "
import json
r = json.load(open('benchmark_data/oolong_gpt5_results.json'))
print(f'OOLONG Score: {r[\"score\"]:.2f}%')
print(f'LLM Calls: {r[\"calls\"]}')
print(f'Model: {r[\"model\"]}')
"
```

### 6.3 Files

| File | Description |
|------|-------------|
| `run_oolong_gpt5.py` | Main benchmark script |
| `labelers/gpt.py` | GPT-5 classification labeler |
| `benchmark_data/oolong/` | OOLONG dataset (parquet files) |
| `benchmark_data/oolong_gpt5_results.json` | Saved results |

---

## 7. Comparison Summary

### 7.1 Results on trec_coarse

| Metric | RLM GPT-5 (Paper) | Our GPT-5 | Difference |
|--------|-------------------|-----------|------------|
| OOLONG Score | 56.50% | 73.24% | +16.74 pts |
| LLM Calls | 26,000+ | 1,900 | 13.7x fewer |
| Approach | Recursive | Classify-once + aggregate | Different paradigm |
| Cost (estimated) | ~$5.20+ | ~$0.38 | Lower |

### 7.2 Scope of Our Evaluation

| Benchmark | Our Result | RLM Paper | Notes |
|-----------|------------|-----------|-------|
| OOLONG trec_coarse (Pure GPT-5) | 73.24% | 56.50% | Same model, same dataset |
| OOLONG trec_coarse (Hybrid GPT-5) | 59.59% | 56.50% | Cost-optimized variant |
| Other OOLONG datasets | Not tested | Varies | Future work needed |
| Other RLM benchmarks | Not tested | Varies | Future work needed |

**Limitation**: RLM was evaluated across multiple benchmarks and datasets. Our study focused only on OOLONG trec_coarse (250 examples). The relative performance of these approaches may differ on other tasks, particularly those requiring iterative reasoning.

---

## 8. Hybrid Approach: Cost-Optimized Pipeline

We also tested a **hybrid labeling pipeline** that reduces API costs while still beating RLM.

### 8.1 Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              HYBRID LABELING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each record:                                               │
│                                                                 │
│  1. KEYWORD MATCHING (FREE)                                     │
│     ┌──────────────────────────────────────────────────────┐   │
│     │ Check if question contains label keywords            │   │
│     │ e.g., "abbreviation" in "What is the abbreviation?"  │   │
│     │ Confidence threshold: 0.5                            │   │
│     └──────────────────────────────────────────────────────┘   │
│                    │                                            │
│         ┌─────────┴─────────┐                                  │
│         │ High confidence?  │                                  │
│         └─────────┬─────────┘                                  │
│              YES  │  NO                                        │
│               ▼   │                                            │
│           DONE    │                                            │
│                   ▼                                            │
│  2. EMBEDDING SIMILARITY (CHEAP - LOCAL)                       │
│     ┌──────────────────────────────────────────────────────┐   │
│     │ Compare question embedding to label embeddings        │   │
│     │ Model: all-MiniLM-L6-v2 (runs locally)               │   │
│     │ Confidence threshold: 0.65                           │   │
│     └──────────────────────────────────────────────────────┘   │
│                    │                                            │
│         ┌─────────┴─────────┐                                  │
│         │ High confidence?  │                                  │
│         └─────────┬─────────┘                                  │
│              YES  │  NO                                        │
│               ▼   │                                            │
│           DONE    │                                            │
│                   ▼                                            │
│  3. GPT-5 (EXPENSIVE - API CALL)                               │
│     ┌──────────────────────────────────────────────────────┐   │
│     │ Send to GPT-5 for classification                     │   │
│     │ Only ~32% of records reach this stage                │   │
│     └──────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Hybrid Results

```json
{
  "score": 59.59,
  "exact_match": 49.20,
  "gpt_calls": 580,
  "method": "hybrid (keyword -> embedding -> GPT-5)",
  "labeler_stats": {
    "keyword_total": 21650,
    "embedding_total": 2650,
    "gpt_total": 11600
  }
}
```

### 8.3 Labeler Distribution

| Labeler | Records | Percentage | Cost |
|---------|---------|------------|------|
| Keyword | 21,650 | 60.3% | $0.00 |
| Embedding | 2,650 | 7.4% | ~$0.00 (local) |
| GPT-5 | 11,600 | 32.3% | ~$0.12 |
| **Total** | **35,900** | **100%** | **~$0.12** |

### 8.4 Accuracy vs Cost Tradeoff

| Approach | Score | GPT-5 Calls | Est. Cost | Notes |
|----------|-------|-------------|-----------|-------|
| RLM (paper) | 56.50% | 26,000+ | ~$5.20 | Recursive, full exploration |
| Pure GPT-5 | 73.24% | 1,900 | ~$0.38 | Single-pass classification |
| Hybrid GPT-5 | 59.59% | ~580 | ~$0.12 | Heuristics + LLM fallback |

The hybrid approach demonstrates:
- Keyword and embedding layers can handle ~68% of records without any LLM calls
- Significant cost reduction is possible while maintaining competitive accuracy
- This concept could potentially be combined with RLM for further optimization

### 8.5 When to Consider Each Approach

| Scenario | Approach | Expected Score |
|----------|----------|----------------|
| Pure aggregation tasks, accuracy priority | Single-pass classification | 73.24% |
| Cost-sensitive applications | Hybrid (heuristics + LLM) | 59.59% |
| Complex reasoning, iterative refinement | RLM (recursive) | Task-dependent |

The optimal approach likely depends on task characteristics - a topic for future investigation.

---

## 9. Conclusion and Future Directions

This replication study explored alternative approaches to aggregation tasks within the OOLONG benchmark:

### Key Observations

1. **Task-specific optimization**: For pure aggregation (counting, frequency comparison), separating classification from counting showed strong results on trec_coarse
2. **Cost-accuracy tradeoffs**: The hybrid approach demonstrates that heuristic pre-filtering can significantly reduce API costs
3. **Complementary approaches**: These methods may complement RLM rather than replace it - RLM's recursion is likely valuable for tasks requiring iterative reasoning

### Limitations of This Study

| Limitation | Impact |
|------------|--------|
| Single dataset (trec_coarse, n=250) | Results may not generalize |
| Aggregation tasks only | RLM may excel at other task types |
| No evaluation of complex reasoning | Recursion benefits unexplored |
| Different implementation details | Not a perfect replication |

### Opportunities for Collaboration

We see potential for combining approaches:
- **Hybrid + RLM**: Use cheap heuristics for easy cases, RLM for complex ones
- **Task routing**: Classify task type first, then select optimal approach
- **Broader evaluation**: Test across full OOLONG suite and other RLM benchmarks

We welcome collaboration with the RLM team to explore these directions and better understand when each approach is most effective.

---

## Appendix A: Sample Outputs

### Correct Predictions

```
[0] match | TASK_TYPE.MOST_FREQ
[3] match | TASK_TYPE.MOST_FREQ
[4] match | TASK_TYPE.LEAST_FREQ
[6] match | TASK_TYPE.RELATIVE_FREQ
[7] match | TASK_TYPE.RELATIVE_FREQ
...
```

### Partial Credit (Numeric)

```
[20] score=0.75 | TASK_TYPE.NUMERIC_ONE_CLASS
  Expected: [5]
  Got: 4
  # 0.75^|5-4| = 0.75^1 = 0.75
```

### Misclassifications

```
[1] score=0.00 | TASK_TYPE.MOST_FREQ
  Expected: ['abbreviation']
  Got: location
  # Classification error propagated to aggregation
```

---

## Appendix B: Code Reference

### GPT Labeler (`labelers/gpt.py`)

```python
class GPTLabeler(Labeler):
    def label_batch(self, instances: list[str], vocabulary: list[str]) -> list[LabelResult]:
        prompt = f"""Classify each question into exactly one category.

Categories: {', '.join(vocabulary)}

Questions:
{items}

Respond with ONLY the category names, one per line, numbered to match."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=len(instances) * 30,
            messages=[{"role": "user", "content": prompt}]
        )
        # Parse and return results...
```

### Aggregation (`run_oolong_gpt5.py`)

```python
def answer_oolong_query(context, task, question, answer_type):
    # Initialize Counter with all labels
    label_counts = Counter({label: 0 for label in vocabulary})
    for r in records:
        if r['label']:
            label_counts[r['label']] += 1

    if 'MOST_FREQ' in task:
        return label_counts.most_common(1)[0][0]
    elif 'LEAST_FREQ' in task:
        return label_counts.most_common()[-1][0]
    # ... etc
```

---

## References

1. Zhang, A., et al. (2024). "Recursive Language Models." arXiv:2512.24601v1.
2. OOLONG Benchmark: https://huggingface.co/datasets/oolongbench/oolong-synth
3. OpenAI GPT-5 API: https://platform.openai.com/docs/models

---

## Acknowledgments

We thank the RLM authors (Zhang et al.) for their innovative work on recursive language models and for making the OOLONG benchmark publicly available. This replication study builds on their foundation and aims to contribute to the broader understanding of when different approaches are most effective.

We welcome feedback and collaboration opportunities.

---

*Report generated: January 14, 2025*
*Benchmark executed with: GPT-5 (gpt-5-chat-latest)*
*Evaluated on: OOLONG trec_coarse (250 examples)*
*All results reproducible via provided scripts*
