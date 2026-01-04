#!/usr/bin/env python3
"""
Full Benchmark Runner

Runs the complete skill-based benchmark with detailed output suitable
for academic reporting.

Usage:
    python3 run_full_benchmark.py
    python3 run_full_benchmark.py --output results.json
"""

import json
import time
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class QueryResult:
    query_id: int
    query_type: str
    question: str
    expected: str
    computed: str
    correct: bool
    time_ms: float
    nodes_visited: int
    edges_traversed: int
    confidence: float
    method: str = "skill-based"


def load_graph(path: str) -> dict:
    """Load knowledge graph."""
    return json.loads(Path(path).read_text())


def load_queries(path: str) -> list:
    """Load benchmark queries."""
    return json.loads(Path(path).read_text())


def execute_query(graph: dict, question: str, query_type: str) -> tuple[str, dict]:
    """
    Execute query using graph traversal.
    Returns (answer, state_info)
    """
    state = {
        "visited_nodes": set(),
        "visited_edges": set(),
        "confidence": 0.0,
    }

    question_lower = question.lower()
    entities = graph["entities"]
    relationships = graph["relationships"]

    if query_type == "count":
        # Find entity by name, return source_docs count
        for eid, entity in entities.items():
            if entity["name"].lower() in question_lower:
                state["visited_nodes"].add(eid)
                state["confidence"] = 1.0
                return str(len(entity["source_docs"])), state
        return "0", state

    elif query_type == "multi_hop":
        # Find person -> traverse works_for -> collect orgs
        match = re.search(r"what companies has (.+?) been involved", question_lower)
        if match:
            person_name = match.group(1)
            for eid, entity in entities.items():
                if entity["name"].lower() == person_name and entity["type"] == "person":
                    state["visited_nodes"].add(eid)
                    companies = set()
                    for rid, rel in relationships.items():
                        if rel["source_entity_id"] == eid and rel["type"] == "works_for":
                            state["visited_edges"].add(rid)
                            target = entities.get(rel["target_entity_id"])
                            if target:
                                state["visited_nodes"].add(rel["target_entity_id"])
                                companies.add(target["name"])
                    state["confidence"] = 1.0
                    return ", ".join(sorted(companies)), state
        return "unknown", state

    elif query_type == "aggregate":
        # Scan edges with category, group by company, return max
        match = re.search(r"which company has the most (\w+)", question_lower)
        if match:
            category = match.group(1)
            company_counts = {}
            for rid, rel in relationships.items():
                if rel.get("attributes", {}).get("category") == category:
                    state["visited_edges"].add(rid)
                    src = entities.get(rel["source_entity_id"])
                    if src and src["type"] == "organization":
                        state["visited_nodes"].add(rel["source_entity_id"])
                        company_counts[src["name"]] = company_counts.get(src["name"], 0) + 1
            if company_counts:
                state["confidence"] = 1.0
                return max(company_counts, key=company_counts.get), state
        return "unknown", state

    return "unknown", state


def check_correct(computed: str, expected: str, query_type: str) -> bool:
    """Check if computed answer matches expected."""
    if query_type == "count":
        return computed == expected
    elif query_type == "multi_hop":
        exp_set = set(x.strip() for x in expected.split(","))
        comp_set = set(x.strip() for x in computed.split(",")) if computed != "unknown" else set()
        return exp_set == comp_set
    else:  # aggregate
        return computed.lower() == expected.lower()


def run_benchmark(graph_path: str, queries_path: str) -> list[QueryResult]:
    """Run full benchmark."""
    graph = load_graph(graph_path)
    queries = load_queries(queries_path)

    results = []

    for i, q in enumerate(queries):
        start = time.perf_counter()
        computed, state = execute_query(graph, q["question"], q["query_type"])
        elapsed = (time.perf_counter() - start) * 1000

        correct = check_correct(computed, q["answer"], q["query_type"])

        result = QueryResult(
            query_id=i + 1,
            query_type=q["query_type"],
            question=q["question"],
            expected=q["answer"],
            computed=computed,
            correct=correct,
            time_ms=elapsed,
            nodes_visited=len(state["visited_nodes"]),
            edges_traversed=len(state["visited_edges"]),
            confidence=state["confidence"],
        )
        results.append(result)

    return results


def print_results(results: list[QueryResult]):
    """Print formatted results."""
    print("=" * 80)
    print("SKILL-BASED KNOWLEDGE GRAPH BENCHMARK RESULTS")
    print("=" * 80)
    print()

    # Detailed results
    print("DETAILED RESULTS:")
    print("-" * 80)
    for r in results:
        status = "✓" if r.correct else "✗"
        print(f"{status} Q{r.query_id:02d} [{r.query_type:10s}] {r.time_ms:6.2f}ms | "
              f"nodes:{r.nodes_visited:3d} edges:{r.edges_traversed:4d} | "
              f"conf:{r.confidence:.0%}")
        if not r.correct:
            print(f"    Expected: {r.expected[:50]}...")
            print(f"    Computed: {r.computed[:50]}...")
    print()

    # Summary by type
    print("SUMMARY BY QUERY TYPE:")
    print("-" * 80)
    by_type = {}
    for r in results:
        if r.query_type not in by_type:
            by_type[r.query_type] = {"correct": 0, "total": 0, "time": 0, "nodes": 0, "edges": 0}
        by_type[r.query_type]["total"] += 1
        by_type[r.query_type]["time"] += r.time_ms
        by_type[r.query_type]["nodes"] += r.nodes_visited
        by_type[r.query_type]["edges"] += r.edges_traversed
        if r.correct:
            by_type[r.query_type]["correct"] += 1

    print(f"{'Type':<12} {'Accuracy':<12} {'Avg Time':<12} {'Avg Nodes':<12} {'Avg Edges':<12}")
    for qtype, stats in by_type.items():
        acc = f"{stats['correct']}/{stats['total']}"
        avg_time = f"{stats['time']/stats['total']:.2f}ms"
        avg_nodes = f"{stats['nodes']/stats['total']:.1f}"
        avg_edges = f"{stats['edges']/stats['total']:.1f}"
        print(f"{qtype:<12} {acc:<12} {avg_time:<12} {avg_nodes:<12} {avg_edges:<12}")
    print()

    # Overall summary
    total_correct = sum(1 for r in results if r.correct)
    total_time = sum(r.time_ms for r in results)
    total_nodes = sum(r.nodes_visited for r in results)
    total_edges = sum(r.edges_traversed for r in results)

    print("OVERALL SUMMARY:")
    print("-" * 80)
    print(f"  Accuracy:      {total_correct}/{len(results)} ({100*total_correct/len(results):.1f}%)")
    print(f"  Total time:    {total_time:.2f}ms")
    print(f"  Avg time:      {total_time/len(results):.2f}ms per query")
    print(f"  LLM calls:     0 (all graph operations)")
    print(f"  Avg nodes:     {total_nodes/len(results):.1f}")
    print(f"  Avg edges:     {total_edges/len(results):.1f}")
    print()

    # Comparison
    print("COMPARISON TO RLM (from paper arXiv:2512.24601v1):")
    print("-" * 80)
    print(f"{'Metric':<20} {'Skill-Based':<20} {'RLM (paper)':<20}")
    print(f"{'Accuracy':<20} {f'{100*total_correct/len(results):.1f}%':<20} {'23-58% F1':<20}")
    print(f"{'Query time':<20} {f'{total_time/len(results):.2f}ms avg':<20} {'seconds-minutes':<20}")
    print(f"{'LLM calls/query':<20} {'0':<20} {'10-1000+':<20}")
    print(f"{'Termination':<20} {'Deterministic':<20} {'Brittle':<20}")
    print(f"{'State tracking':<20} {'Yes':<20} {'No':<20}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark")
    parser.add_argument("--graph", "-g", default="benchmark_data/graph_structured.json")
    parser.add_argument("--queries", "-q", default="benchmark_data/queries.json")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    args = parser.parse_args()

    # Check files exist
    if not Path(args.graph).exists():
        print(f"Error: {args.graph} not found")
        print("Run 'python3 build_graph.py' first")
        return 1

    if not Path(args.queries).exists():
        print(f"Error: {args.queries} not found")
        print("Run 'python3 generate_benchmark.py' first")
        return 1

    # Run benchmark
    results = run_benchmark(args.graph, args.queries)

    # Print results
    print_results(results)

    # Save if requested
    if args.output:
        output_data = {
            "results": [asdict(r) for r in results],
            "summary": {
                "total": len(results),
                "correct": sum(1 for r in results if r.correct),
                "accuracy": sum(1 for r in results if r.correct) / len(results),
                "total_time_ms": sum(r.time_ms for r in results),
                "avg_time_ms": sum(r.time_ms for r in results) / len(results),
            }
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
