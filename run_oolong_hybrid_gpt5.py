#!/usr/bin/env python3
"""
OOLONG Benchmark with Hybrid GPT-5 Labeler (Keyword -> Embedding -> GPT-5)

Apples-to-apples comparison with RLM paper (arXiv:2512.24601).

This script uses:
- context_window_text (WITHOUT labels) per MIT feedback
- Hybrid labeler: keyword (free) -> embedding (cheap) -> GPT-5 (expensive)
- Deterministic aggregation via Counter
- trec_coarse dataset only per MIT recommendation
- OOLONG metric: 0.75^|y-ŷ| for numeric, exact match for categorical
"""

import os
import re
import json
import time
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd

from labelers.hybrid_gpt import HybridGPTLabeler, HybridGPTLabelerConfig


@dataclass
class OolongContext:
    """Parsed OOLONG context with structured records."""
    context_id: int
    dataset: str
    records: list[dict]
    labels: list[str]


def parse_context_unlabeled(context_text: str, context_id: int, dataset: str) -> OolongContext:
    """Parse OOLONG context WITHOUT ground-truth labels."""
    records = []
    labels = []

    label_match = re.search(r"one of (\d+) categories: (.+?)\.?\n", context_text)
    if label_match:
        label_str = label_match.group(2)
        labels = re.findall(r"'([^']+)'", label_str)

    record_pattern = r"Date:\s*([^|]+)\|\|\s*User:\s*(\d+)\s*\|\|\s*Instance:\s*([^|\n]+)"

    for match in re.finditer(record_pattern, context_text):
        records.append({
            'date': match.group(1).strip(),
            'user': match.group(2).strip(),
            'instance': match.group(3).strip(),
            'label': None,
            'confidence': 0.0,
        })

    return OolongContext(
        context_id=context_id,
        dataset=dataset,
        records=records,
        labels=labels
    )


def answer_oolong_query(context: OolongContext, task: str, question: str, answer_type: str) -> str:
    """Answer OOLONG query using deterministic aggregation."""
    records = [r for r in context.records if r.get('label')]
    vocabulary = context.labels

    if not records:
        return "No labeled records"

    label_counts = Counter({label: 0 for label in vocabulary})
    for r in records:
        if r['label']:
            label_counts[r['label']] += 1

    if 'MOST_FREQ' in task and 'SECOND' not in task:
        if 'USER' in answer_type:
            user_counts = Counter(r['user'] for r in records)
            if user_counts:
                return user_counts.most_common(1)[0][0]
        else:
            if label_counts:
                return label_counts.most_common(1)[0][0]

    elif 'SECOND_MOST_FREQ' in task:
        if 'USER' in answer_type:
            user_counts = Counter(r['user'] for r in records)
            if len(user_counts) >= 2:
                return user_counts.most_common(2)[1][0]
        else:
            if len(label_counts) >= 2:
                return label_counts.most_common(2)[1][0]

    elif 'LEAST_FREQ' in task:
        if 'USER' in answer_type:
            user_counts = Counter(r['user'] for r in records)
            if user_counts:
                return user_counts.most_common()[-1][0]
        else:
            if label_counts:
                return label_counts.most_common()[-1][0]

    elif 'RELATIVE_FREQ' in task:
        label1_match = re.search(r"label '([^']+)'", question)
        label2_match = re.search(r"as label '([^']+)'", question)
        if label1_match and label2_match:
            label1 = label1_match.group(1)
            label2 = label2_match.group(1)
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
            count = label_counts.get(target_label, 0)
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
                matching = [l for l, c in label_counts.items() if c == n]
                return matching if matching else []

    return "Unknown"


def normalize_answer(answer) -> Union[str, list]:
    """Normalize answer for comparison."""
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


def oolong_score(predicted, expected) -> float:
    """Calculate OOLONG score: 0.75^|y-ŷ| for numeric, exact match for categorical."""
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    try:
        pred_num = float(pred_norm) if isinstance(pred_norm, str) else None
        exp_num = float(exp_norm) if isinstance(exp_norm, str) else None

        if pred_num is not None and exp_num is not None:
            return 0.75 ** abs(exp_num - pred_num)
    except (ValueError, TypeError):
        pass

    if isinstance(exp_norm, list):
        if isinstance(pred_norm, list):
            return 1.0 if set(pred_norm) == set(exp_norm) else 0.0
        return 1.0 if pred_norm in exp_norm else 0.0

    return 1.0 if pred_norm == exp_norm else 0.0


def run_benchmark(
    parquet_dir: str,
    gpt_model: str = "gpt-5-chat-latest",
    dataset_filter: str = "trec_coarse",
    verbose: bool = True
):
    """Run OOLONG benchmark with Hybrid GPT-5 Labeler."""

    # Load data
    dfs = []
    for f in os.listdir(parquet_dir):
        if f.endswith('.parquet'):
            dfs.append(pd.read_parquet(os.path.join(parquet_dir, f)))

    df = pd.concat(dfs, ignore_index=True)

    if dataset_filter:
        df = df[df['dataset'] == dataset_filter]

    if verbose:
        print("=" * 70)
        print(f"OOLONG BENCHMARK - HYBRID GPT-5")
        print(f"Pipeline: Keyword -> Embedding -> GPT-5 ({gpt_model})")
        print("=" * 70)
        print(f"\nDataset: {dataset_filter}")
        print(f"Total examples: {len(df)}")
        print(f"Task types: {df['task'].nunique()}")

    # Initialize hybrid GPT-5 labeler
    config = HybridGPTLabelerConfig(gpt_model=gpt_model)
    labeler = HybridGPTLabeler(config)

    results = {
        'total': 0,
        'score_sum': 0.0,
        'exact_match': 0,
        'by_task': defaultdict(lambda: {'score_sum': 0.0, 'total': 0}),
        'times': [],
        'errors': [],
        'labeler_stats': {
            'keyword_total': 0,
            'embedding_total': 0,
            'gpt_total': 0,
        }
    }

    for idx, row in df.iterrows():
        start_time = time.time()

        # Reset labeler stats for each query
        labeler.reset_stats()

        try:
            context = parse_context_unlabeled(
                row['context_window_text'],
                row['context_window_id'],
                row['dataset']
            )

            if not context.records:
                results['errors'].append((idx, "No records"))
                continue

            if not context.labels:
                results['errors'].append((idx, "No vocabulary"))
                continue

            # Label with Hybrid GPT-5
            labeler.label_records(context.records, context.labels)

            # Track labeler stats
            results['labeler_stats']['keyword_total'] += labeler.stats.keyword_hits
            results['labeler_stats']['embedding_total'] += labeler.stats.embedding_hits
            results['labeler_stats']['gpt_total'] += labeler.stats.gpt_records_labeled

            # Aggregate
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

        score = oolong_score(predicted, row['answer'])
        results['score_sum'] += score
        results['total'] += 1

        if score == 1.0:
            results['exact_match'] += 1

        results['by_task'][row['task']]['score_sum'] += score
        results['by_task'][row['task']]['total'] += 1

        if verbose:
            status = "match" if score == 1.0 else f"score={score:.2f}"
            print(f"[{idx}] {status} | {row['task'][:20]} | kw:{labeler.stats.keyword_hits} emb:{labeler.stats.embedding_hits} gpt:{labeler.stats.gpt_records_labeled}")
            if score < 1.0:
                print(f"  Expected: {row['answer']}")
                print(f"  Got: {predicted}")

    # Get final GPT call count
    total_gpt_calls = labeler.gpt.call_count if labeler._gpt else 0

    # Summary
    avg_score = results['score_sum'] / results['total'] * 100 if results['total'] else 0
    exact_rate = results['exact_match'] / results['total'] * 100 if results['total'] else 0
    avg_time = sum(results['times']) / len(results['times']) * 1000 if results['times'] else 0

    # Calculate labeler distribution
    total_records = (results['labeler_stats']['keyword_total'] +
                    results['labeler_stats']['embedding_total'] +
                    results['labeler_stats']['gpt_total'])

    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nOOLONG Score: {avg_score:.2f}%")
        print(f"Exact Match: {exact_rate:.2f}%")
        print(f"Total GPT-5 API calls: {total_gpt_calls}")
        print(f"Avg time/query: {avg_time:.1f}ms")

        print(f"\nLabeler Distribution:")
        if total_records > 0:
            kw_pct = results['labeler_stats']['keyword_total'] / total_records * 100
            emb_pct = results['labeler_stats']['embedding_total'] / total_records * 100
            gpt_pct = results['labeler_stats']['gpt_total'] / total_records * 100
            print(f"  Keyword:   {results['labeler_stats']['keyword_total']:5d} ({kw_pct:.1f}%)")
            print(f"  Embedding: {results['labeler_stats']['embedding_total']:5d} ({emb_pct:.1f}%)")
            print(f"  GPT-5:     {results['labeler_stats']['gpt_total']:5d} ({gpt_pct:.1f}%)")

        print(f"\nBy Task:")
        for task, stats in sorted(results['by_task'].items()):
            task_score = stats['score_sum'] / stats['total'] * 100 if stats['total'] else 0
            print(f"  {task}: {task_score:.1f}%")

        print("\n" + "=" * 70)
        print("COMPARISON TO RLM PAPER (arXiv:2512.24601)")
        print("=" * 70)
        print(f"{'Method':<40} {'Score':<12} {'LLM Calls':<10}")
        print(f"{'RLM GPT-5 (paper)':<40} {'56.50%':<12} {'26000+':<10}")
        print(f"{'Our Pure GPT-5':<40} {'73.24%':<12} {'1900':<10}")
        print(f"{'Our Hybrid (kw+emb+GPT-5)':<40} {f'{avg_score:.2f}%':<12} {f'{total_gpt_calls}':<10}")

    return {
        'score': avg_score,
        'exact_match': exact_rate,
        'gpt_calls': total_gpt_calls,
        'gpt_model': gpt_model,
        'method': 'hybrid (keyword -> embedding -> GPT-5)',
        'labeler_stats': results['labeler_stats'],
        'by_task': {k: {'score': v['score_sum']/v['total']*100 if v['total'] else 0, 'n': v['total']}
                   for k, v in results['by_task'].items()}
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OOLONG with Hybrid GPT-5 Labeler")
    parser.add_argument("--parquet-dir", "-d", default="benchmark_data/oolong")
    parser.add_argument("--gpt-model", "-m", default="gpt-5-chat-latest")
    parser.add_argument("--dataset", default="trec_coarse")
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()

    results = run_benchmark(
        args.parquet_dir,
        gpt_model=args.gpt_model,
        dataset_filter=args.dataset,
        verbose=not args.quiet
    )

    # Save results
    output_path = Path("benchmark_data/oolong_hybrid_gpt5_results.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")
