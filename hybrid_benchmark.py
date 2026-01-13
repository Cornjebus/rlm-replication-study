#!/usr/bin/env python3
"""
Hybrid OOLONG Benchmark

Tests the hybrid approach:
1. Parse records WITHOUT labels (addressing MIT feedback)
2. Cheap-first labeling (keyword → embedding → RLM)
3. Deterministic aggregation (Counter operations)

This proves: "Recursion only needed for semantic inference, not aggregation"
"""

import os
import re
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from labelers import HybridLabeler, HybridLabelerConfig, HybridLabelerStats


@dataclass
class OolongContext:
    """Parsed OOLONG context with structured records."""
    context_id: int
    dataset: str
    records: list[dict]  # Each record: {date, user, instance, label, confidence}
    labels: list[str]    # Available label vocabulary


def parse_context_unlabeled(context_text: str, context_id: int, dataset: str) -> OolongContext:
    """
    Parse OOLONG context WITHOUT using ground-truth labels.

    This is the key difference from the original benchmark - we don't use
    context_window_text_with_labels. Labels must be inferred.
    """
    records = []
    labels = []

    # Extract available labels from intro (vocabulary only, not answers)
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
            'label': None,      # Must be inferred
            'confidence': 0.0,
            'label_source': None
        })

    return OolongContext(
        context_id=context_id,
        dataset=dataset,
        records=records,
        labels=labels
    )


def answer_oolong_query(context: OolongContext, task: str, question: str, answer_type: str) -> str:
    """
    Answer OOLONG query using deterministic aggregation.

    This is the same aggregation logic as before - Counter operations.
    The difference is that labels were inferred, not given.
    """
    records = context.records

    # Filter to records with labels (some may have failed inference)
    records = [r for r in records if r.get('label')]

    if not records:
        return "No labeled records"

    # Handle different task types with Counter operations
    if 'MOST_FREQ' in task and 'SECOND' not in task:
        if 'USER' in answer_type:
            user_counts = Counter(r['user'] for r in records)
            if user_counts:
                return user_counts.most_common(1)[0][0]
        else:
            label_counts = Counter(r['label'] for r in records)
            if label_counts:
                return label_counts.most_common(1)[0][0]

    elif 'SECOND_MOST_FREQ' in task:
        if 'USER' in answer_type:
            user_counts = Counter(r['user'] for r in records)
            if len(user_counts) >= 2:
                return user_counts.most_common(2)[1][0]
        else:
            label_counts = Counter(r['label'] for r in records)
            if len(label_counts) >= 2:
                return label_counts.most_common(2)[1][0]

    elif 'LEAST_FREQ' in task:
        if 'USER' in answer_type:
            user_counts = Counter(r['user'] for r in records)
            if user_counts:
                return user_counts.most_common()[-1][0]
        else:
            label_counts = Counter(r['label'] for r in records)
            if label_counts:
                return label_counts.most_common()[-1][0]

    elif 'RELATIVE_FREQ' in task:
        # Compare two labels
        label1_match = re.search(r"label '([^']+)'", question)
        label2_match = re.search(r"as label '([^']+)'", question)
        if label1_match and label2_match:
            label1 = label1_match.group(1)
            label2 = label2_match.group(1)
            label_counts = Counter(r['label'] for r in records)
            count1 = label_counts.get(label1, 0)
            count2 = label_counts.get(label2, 0)
            if count1 > count2:
                return "more common than"
            elif count1 < count2:
                return "less common than"
            else:
                return "same frequency as"

    elif 'NUMERIC_ONE_CLASS' in task:
        label_match = re.search(r"label '([^']+)'", question)
        if label_match:
            target_label = label_match.group(1)
            count = sum(1 for r in records if r['label'] == target_label)
            return str(count)

    elif 'REPRESENTED_N_TIMES' in task:
        n_match = re.search(r"exactly (\d+) times", question)
        if n_match:
            n = int(n_match.group(1))
            if 'USER' in answer_type:
                user_counts = Counter(r['user'] for r in records)
                matching = [u for u, c in user_counts.items() if c == n]
                return matching if matching else []
            else:
                label_counts = Counter(r['label'] for r in records)
                matching = [l for l, c in label_counts.items() if c == n]
                return matching if matching else []

    return "Unknown"


def normalize_answer(answer):
    """Normalize answer for comparison."""
    import ast

    if isinstance(answer, list):
        if len(answer) == 1:
            return str(answer[0]).strip().lower()
        return [str(a).strip().lower() for a in answer]

    answer_str = str(answer).strip()
    if answer_str.startswith('[') and answer_str.endswith(']'):
        try:
            parsed = ast.literal_eval(answer_str)
            if isinstance(parsed, list):
                if len(parsed) == 1:
                    return str(parsed[0]).strip().lower()
                return [str(a).strip().lower() for a in parsed]
        except:
            pass

    return answer_str.lower()


def check_answer(predicted, expected):
    """Check if predicted answer matches expected."""
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    if isinstance(exp_norm, list):
        if isinstance(pred_norm, list):
            return set(pred_norm) == set(exp_norm)
        return pred_norm in exp_norm
    return pred_norm == exp_norm


def run_hybrid_benchmark(
    parquet_dir: str,
    config: Optional[HybridLabelerConfig] = None,
    verbose: bool = True,
    dataset_filter: str = "trec_coarse"  # Per MIT, focus on this
):
    """
    Run hybrid benchmark on OOLONG data.

    Args:
        parquet_dir: Directory containing OOLONG parquet files
        config: Hybrid labeler configuration
        verbose: Print detailed output
        dataset_filter: Dataset to evaluate on (default: trec_coarse per MIT)
    """
    config = config or HybridLabelerConfig()

    # Load all parquet files
    dfs = []
    for f in os.listdir(parquet_dir):
        if f.endswith('.parquet'):
            dfs.append(pd.read_parquet(os.path.join(parquet_dir, f)))

    df = pd.concat(dfs, ignore_index=True)

    # Filter to specified dataset (MIT recommends trec_coarse only)
    if dataset_filter:
        df = df[df['dataset'] == dataset_filter]

    if verbose:
        print("=" * 70)
        print("HYBRID OOLONG BENCHMARK")
        print("=" * 70)
        print(f"\nDataset: {dataset_filter}")
        print(f"Total examples: {len(df)}")
        print(f"Task types: {df['task'].nunique()}")
        print(f"\nConfig:")
        print(f"  Keyword threshold: {config.keyword_threshold}")
        print(f"  Embedding threshold: {config.embedding_threshold}")
        print(f"  Max RLM calls/query: {config.max_rlm_calls_per_query}")
        print(f"  RLM batch size: {config.batch_size}")

    results = {
        'total': 0,
        'correct': 0,
        'zero_rlm_queries': 0,
        'total_rlm_calls': 0,
        'total_records': 0,
        'rlm_records': 0,
        'by_source': {'keyword': 0, 'embedding': 0, 'rlm': 0},
        'by_task': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'times': [],
        'errors': []
    }

    for idx, row in df.iterrows():
        start_time = time.time()

        try:
            # 1. PARSE (no labels)
            context = parse_context_unlabeled(
                row['context_window_text'],  # NOT with_labels!
                row['context_window_id'],
                row['dataset']
            )

            if not context.records:
                results['errors'].append((idx, "No records parsed"))
                continue

            if not context.labels:
                results['errors'].append((idx, "No vocabulary found"))
                continue

            # 2. LABEL (hybrid: keyword → embedding → RLM)
            labeler = HybridLabeler(config)
            labeled_records = labeler.label_records(context.records, context.labels)
            context.records = labeled_records

            # Track labeler stats
            results['total_rlm_calls'] += labeler.stats.rlm_calls
            results['total_records'] += labeler.stats.total_records
            results['rlm_records'] += labeler.stats.rlm_records_labeled
            results['by_source']['keyword'] += labeler.stats.keyword_hits
            results['by_source']['embedding'] += labeler.stats.embedding_hits
            results['by_source']['rlm'] += labeler.stats.rlm_records_labeled

            if labeler.stats.zero_rlm:
                results['zero_rlm_queries'] += 1

            # 3. AGGREGATE (deterministic Counter operations)
            predicted = answer_oolong_query(
                context,
                row['task'],
                row['question'],
                row['answer_type']
            )

        except Exception as e:
            predicted = f"Error: {e}"
            results['errors'].append((idx, str(e)))

        elapsed = time.time() - start_time
        results['times'].append(elapsed)

        # Check answer
        is_correct = check_answer(predicted, row['answer'])

        results['total'] += 1
        if is_correct:
            results['correct'] += 1

        results['by_task'][row['task']]['total'] += 1
        if is_correct:
            results['by_task'][row['task']]['correct'] += 1

        if verbose and (not is_correct or idx < 5):
            status = "correct" if is_correct else "WRONG"
            print(f"\n[{idx}] {status}")
            print(f"  Task: {row['task']}")
            print(f"  Records: {len(context.records)}")
            print(f"  RLM calls: {labeler.stats.rlm_calls}")
            print(f"  Question: {row['question'][:80]}...")
            print(f"  Expected: {row['answer']}")
            print(f"  Predicted: {predicted}")

    # Summary
    accuracy = results['correct'] / results['total'] * 100 if results['total'] else 0
    zero_rlm_rate = results['zero_rlm_queries'] / results['total'] * 100 if results['total'] else 0
    avg_rlm = results['total_rlm_calls'] / results['total'] if results['total'] else 0
    avg_time = sum(results['times']) / len(results['times']) * 1000 if results['times'] else 0

    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"\nAccuracy: {results['correct']}/{results['total']} ({accuracy:.1f}%)")
        print(f"Zero-RLM queries: {results['zero_rlm_queries']}/{results['total']} ({zero_rlm_rate:.1f}%)")
        print(f"Avg RLM calls/query: {avg_rlm:.2f}")
        print(f"Avg time/query: {avg_time:.1f}ms")

        print(f"\nLabeling breakdown:")
        total = sum(results['by_source'].values())
        for source, count in results['by_source'].items():
            pct = count / total * 100 if total else 0
            print(f"  {source}: {count} ({pct:.1f}%)")

        print(f"\nBy Task Type:")
        for task, stats in sorted(results['by_task'].items()):
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] else 0
            print(f"  {task}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

        if results['errors']:
            print(f"\nErrors: {len(results['errors'])}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run hybrid OOLONG benchmark")
    parser.add_argument("--parquet-dir", "-d",
                       default="benchmark_data/oolong",
                       help="Directory with OOLONG parquet files")
    parser.add_argument("--dataset", "-ds",
                       default="trec_coarse",
                       help="Dataset to evaluate (default: trec_coarse)")
    parser.add_argument("--keyword-threshold", "-kt",
                       type=float, default=0.5,
                       help="Keyword confidence threshold")
    parser.add_argument("--embedding-threshold", "-et",
                       type=float, default=0.65,
                       help="Embedding confidence threshold")
    parser.add_argument("--max-rlm-calls", "-m",
                       type=int, default=10,
                       help="Max RLM calls per query")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimal output")

    args = parser.parse_args()

    config = HybridLabelerConfig(
        keyword_threshold=args.keyword_threshold,
        embedding_threshold=args.embedding_threshold,
        max_rlm_calls_per_query=args.max_rlm_calls
    )

    results = run_hybrid_benchmark(
        args.parquet_dir,
        config=config,
        verbose=not args.quiet,
        dataset_filter=args.dataset
    )

    # Save results
    output = {
        'dataset': args.dataset,
        'total': results['total'],
        'correct': results['correct'],
        'accuracy': results['correct'] / results['total'] if results['total'] else 0,
        'zero_rlm_queries': results['zero_rlm_queries'],
        'zero_rlm_rate': results['zero_rlm_queries'] / results['total'] if results['total'] else 0,
        'total_rlm_calls': results['total_rlm_calls'],
        'avg_rlm_calls': results['total_rlm_calls'] / results['total'] if results['total'] else 0,
        'avg_time_ms': sum(results['times']) / len(results['times']) * 1000 if results['times'] else 0,
        'by_source': results['by_source'],
        'by_task': {k: dict(v) for k, v in results['by_task'].items()},
        'config': {
            'keyword_threshold': args.keyword_threshold,
            'embedding_threshold': args.embedding_threshold,
            'max_rlm_calls': args.max_rlm_calls
        }
    }

    output_path = Path("benchmark_data/hybrid_results.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
