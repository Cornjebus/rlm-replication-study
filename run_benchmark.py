#!/usr/bin/env python3
"""
Benchmark Runner: RLM vs Skill-Based Knowledge Graph

Runs both approaches on the same queries and compares:
- Accuracy
- Cost (LLM API calls)
- Termination behavior (key difference)

Usage:
    # First generate benchmark and index:
    python3 generate_benchmark.py --records 1000 --queries 20
    python3 recursive-knowledge/scripts/index_corpus.py \
        --input benchmark_data/corpus --output benchmark_data/graph.json -v
    
    # Then run comparison:
    python3 run_benchmark.py \
        --graph benchmark_data/graph.json \
        --queries benchmark_data/queries.json \
        --methods skill rlm
"""

import json
import time
import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# Add skill scripts to path
sys.path.insert(0, str(Path(__file__).parent / "recursive-knowledge" / "scripts"))


# =============================================================================
# RESULT TRACKING
# =============================================================================

@dataclass
class RunResult:
    query_id: str
    method: str
    question: str
    expected: str
    actual: str
    correct: bool
    
    # Performance
    time_seconds: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    
    # Skill-specific
    nodes_visited: Optional[int] = None
    edges_visited: Optional[int] = None
    confidence: Optional[float] = None
    termination_reason: Optional[str] = None
    
    # RLM-specific
    iterations: Optional[int] = None


# =============================================================================
# SKILL-BASED APPROACH
# =============================================================================

def run_skill_approach(query: dict, graph_path: Path, verbose: bool = False) -> RunResult:
    """Run query using skill-based knowledge graph."""
    from graph_ops import KnowledgeGraph
    from query import execute_query
    
    start = time.time()
    
    graph = KnowledgeGraph.load(str(graph_path))
    
    result = execute_query(
        graph,
        query["question"],
        max_depth=5,
        confidence_threshold=0.85,
        min_sources=3,
        verbose=verbose,
    )
    
    elapsed = time.time() - start
    
    # Check correctness
    expected = query["answer"].lower().strip()
    actual = result.get("answer", "").lower().strip()
    correct = expected in actual or actual in expected
    
    # For numeric answers, try exact match
    if query.get("query_type") == "count":
        try:
            correct = str(int(expected)) in actual
        except:
            pass
    
    return RunResult(
        query_id=query["id"],
        method="skill",
        question=query["question"],
        expected=query["answer"],
        actual=result.get("answer", ""),
        correct=correct,
        time_seconds=elapsed,
        llm_calls=1,  # Graph traversal, no LLM calls during query
        input_tokens=0,
        output_tokens=0,
        nodes_visited=result.get("nodes_visited", 0),
        edges_visited=result.get("edges_visited", 0),
        confidence=result.get("confidence", 0),
        termination_reason=result.get("termination_reason", ""),
    )


# =============================================================================
# RLM APPROACH (Simulated)
# =============================================================================

RLM_SYSTEM = """You have access to a corpus loaded as 'context'. 
Use the REPL to explore it and find the answer.
When done, output FINAL(your answer)."""

def run_rlm_approach(query: dict, corpus_path: Path, verbose: bool = False) -> RunResult:
    """
    Run query using RLM approach (simulated).
    
    This simulates what RLM does:
    - Load full corpus as context variable
    - Let model write code to explore
    - No state tracking, no termination conditions
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
    except:
        # Return simulated result if no API key
        return RunResult(
            query_id=query["id"],
            method="rlm",
            question=query["question"],
            expected=query["answer"],
            actual="[API_KEY_REQUIRED]",
            correct=False,
            time_seconds=0,
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            iterations=0,
        )
    
    start = time.time()
    
    # Load corpus (RLM loads everything into context)
    corpus_file = corpus_path / "corpus.json"
    if corpus_file.exists():
        corpus_data = json.loads(corpus_file.read_text())
    else:
        # Load individual files
        corpus_data = []
        for f in corpus_path.glob("*.txt"):
            corpus_data.append({"file": f.name, "content": f.read_text()})
    
    # Truncate to fit context (RLM's problem)
    corpus_str = json.dumps(corpus_data[:100])  # Limit for demo
    
    total_input = 0
    total_output = 0
    iterations = 0
    max_iterations = 10
    
    messages = []
    user_msg = f"""Query: {query['question']}

The corpus has {len(corpus_data)} records. Here's a sample:
{corpus_str[:5000]}...

Write Python code to find the answer. Use FINAL(answer) when done."""

    while iterations < max_iterations:
        iterations += 1
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=RLM_SYSTEM,
            messages=messages + [{"role": "user", "content": user_msg}]
        )
        
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens
        
        text = response.content[0].text
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": text})
        
        if verbose:
            print(f"    Iteration {iterations}: {len(text)} chars")
        
        # Check for FINAL
        if "FINAL(" in text:
            import re
            match = re.search(r'FINAL\(([^)]+)\)', text)
            if match:
                answer = match.group(1).strip()
                break
        
        user_msg = "[Code executed. Continue or use FINAL(answer).]"
    else:
        answer = "MAX_ITERATIONS"
    
    elapsed = time.time() - start
    
    expected = query["answer"].lower().strip()
    actual = answer.lower().strip()
    correct = expected in actual or actual in expected
    
    return RunResult(
        query_id=query["id"],
        method="rlm",
        question=query["question"],
        expected=query["answer"],
        actual=answer,
        correct=correct,
        time_seconds=elapsed,
        llm_calls=iterations,
        input_tokens=total_input,
        output_tokens=total_output,
        iterations=iterations,
    )


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_benchmark(
    graph_path: Path,
    queries_path: Path,
    corpus_path: Path,
    methods: list[str],
    n_queries: int = None,
    verbose: bool = False,
):
    """Run full benchmark comparison."""
    
    # Load queries
    queries = json.loads(queries_path.read_text())
    if n_queries:
        queries = queries[:n_queries]
    
    print("=" * 70)
    print("RLM vs SKILL-BASED KNOWLEDGE GRAPH BENCHMARK")
    print("=" * 70)
    print(f"Queries: {len(queries)}")
    print(f"Methods: {methods}")
    print(f"Graph: {graph_path}")
    print()
    
    results = {m: [] for m in methods}
    
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] {query['question'][:60]}...")
        print(f"    Expected: {query['answer']}")
        
        if "skill" in methods:
            print("    Running SKILL...")
            try:
                r = run_skill_approach(query, graph_path, verbose)
                results["skill"].append(asdict(r))
                status = "✓" if r.correct else "✗"
                print(f"    SKILL: {status} '{r.actual}' (conf: {r.confidence:.0%}, nodes: {r.nodes_visited})")
            except Exception as e:
                print(f"    SKILL: Error - {e}")
        
        if "rlm" in methods:
            print("    Running RLM...")
            try:
                r = run_rlm_approach(query, corpus_path, verbose)
                results["rlm"].append(asdict(r))
                status = "✓" if r.correct else "✗"
                print(f"    RLM: {status} '{r.actual}' (iters: {r.iterations}, tokens: {r.input_tokens})")
            except Exception as e:
                print(f"    RLM: Error - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for method, method_results in results.items():
        if not method_results:
            continue
        
        n_correct = sum(1 for r in method_results if r["correct"])
        total_time = sum(r["time_seconds"] for r in method_results)
        total_calls = sum(r["llm_calls"] for r in method_results)
        total_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in method_results)
        
        print(f"\n{method.upper()}:")
        print(f"  Accuracy: {n_correct}/{len(method_results)} ({100*n_correct/len(method_results):.1f}%)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  LLM calls: {total_calls}")
        print(f"  Total tokens: {total_tokens:,}")
        
        if method == "skill":
            avg_conf = sum(r.get("confidence", 0) or 0 for r in method_results) / len(method_results)
            avg_nodes = sum(r.get("nodes_visited", 0) or 0 for r in method_results) / len(method_results)
            print(f"  Avg confidence: {avg_conf:.0%}")
            print(f"  Avg nodes visited: {avg_nodes:.1f}")
        
        if method == "rlm":
            avg_iters = sum(r.get("iterations", 0) or 0 for r in method_results) / len(method_results)
            print(f"  Avg iterations: {avg_iters:.1f}")
    
    # Save results
    results_file = Path(f"benchmark_results_{int(time.time())}.json")
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run benchmark comparison")
    parser.add_argument("--graph", required=True, help="Path to knowledge graph JSON")
    parser.add_argument("--queries", required=True, help="Path to queries JSON")
    parser.add_argument("--corpus", default="./benchmark_data/corpus", help="Path to corpus directory")
    parser.add_argument("--methods", nargs="+", default=["skill"], choices=["skill", "rlm"])
    parser.add_argument("--n", type=int, help="Number of queries to run")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    run_benchmark(
        graph_path=Path(args.graph),
        queries_path=Path(args.queries),
        corpus_path=Path(args.corpus),
        methods=args.methods,
        n_queries=args.n,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
