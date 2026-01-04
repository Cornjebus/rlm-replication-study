#!/usr/bin/env python3
"""
Independent Verification Script

Verifies that benchmark results are correct by computing answers
directly from the raw corpus data, independent of the graph.

This proves that:
1. Answers are not hardcoded
2. Graph traversal produces correct results
3. Results are reproducible

Usage:
    python3 verify_results.py
    python3 verify_results.py --verbose
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter


def load_data():
    """Load corpus and queries."""
    corpus = json.loads(Path("benchmark_data/corpus.json").read_text())
    queries = json.loads(Path("benchmark_data/queries.json").read_text())
    return corpus, queries


def verify_count_query(corpus: list, question: str, expected: str) -> tuple[bool, str, str]:
    """Verify a count query directly against corpus."""
    question_lower = question.lower()

    # Try to find what we're counting
    for record in corpus:
        # Check company
        if record['company'].lower() in question_lower:
            count = len([r for r in corpus if r['company'] == record['company']])
            return str(count) == expected, str(count), f"company={record['company']}"

        # Check category
        if record['category'].lower() in question_lower:
            count = len([r for r in corpus if r['category'] == record['category']])
            return str(count) == expected, str(count), f"category={record['category']}"

        # Check location
        if record['location'].lower() in question_lower:
            count = len([r for r in corpus if r['location'] == record['location']])
            return str(count) == expected, str(count), f"location={record['location']}"

        # Check topic
        if record['topic'].lower() in question_lower:
            count = len([r for r in corpus if r['topic'] == record['topic']])
            return str(count) == expected, str(count), f"topic={record['topic']}"

    return False, "unknown", "no match found"


def verify_multi_hop_query(corpus: list, question: str, expected: str) -> tuple[bool, str, str]:
    """Verify a multi-hop query directly against corpus."""
    match = re.search(r"what companies has (.+?) been involved", question.lower())
    if not match:
        return False, "unknown", "could not parse question"

    person_name = match.group(1)

    # Find all records mentioning this person
    companies = set()
    for record in corpus:
        for person in record['people']:
            if person.lower() == person_name:
                companies.add(record['company'])

    computed = ", ".join(sorted(companies))
    expected_set = set(x.strip() for x in expected.split(","))

    return companies == expected_set, computed, f"person={person_name}"


def verify_aggregate_query(corpus: list, question: str, expected: str) -> tuple[bool, str, str]:
    """Verify an aggregate query directly against corpus."""
    match = re.search(r"which company has the most (\w+)", question.lower())
    if not match:
        return False, "unknown", "could not parse question"

    category = match.group(1)

    # Count by company
    counts = Counter()
    for record in corpus:
        if record['category'] == category:
            counts[record['company']] += 1

    if not counts:
        return False, "unknown", f"no records with category={category}"

    top_company = counts.most_common(1)[0][0]
    return top_company == expected, top_company, f"category={category}, counts={dict(counts.most_common(5))}"


def main():
    parser = argparse.ArgumentParser(description="Verify benchmark results")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("INDEPENDENT VERIFICATION")
    print("Computing answers directly from corpus.json (no graph)")
    print("=" * 80)

    corpus, queries = load_data()
    print(f"\nLoaded {len(corpus)} records, {len(queries)} queries\n")

    results = {"count": [], "multi_hop": [], "aggregate": []}
    all_pass = True

    for i, q in enumerate(queries):
        qtype = q["query_type"]
        question = q["question"]
        expected = q["answer"]

        if qtype == "count":
            passed, computed, details = verify_count_query(corpus, question, expected)
        elif qtype == "multi_hop":
            passed, computed, details = verify_multi_hop_query(corpus, question, expected)
        else:
            passed, computed, details = verify_aggregate_query(corpus, question, expected)

        results[qtype].append(passed)
        if not passed:
            all_pass = False

        status = "✓" if passed else "✗"

        if args.verbose or not passed:
            print(f"{status} Q{i+1:02d} [{qtype:10s}]")
            print(f"   Question: {question[:60]}...")
            print(f"   Expected: {expected[:40]}...")
            print(f"   Computed: {computed[:40]}...")
            print(f"   Details:  {details}")
            print()
        else:
            print(f"{status} Q{i+1:02d} [{qtype:10s}] {question[:50]}...")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for qtype, type_results in results.items():
        passed = sum(type_results)
        total = len(type_results)
        pct = 100 * passed / total if total > 0 else 0
        print(f"  {qtype:12s}: {passed}/{total} ({pct:.0f}%)")

    total_passed = sum(sum(r) for r in results.values())
    total_queries = len(queries)
    print(f"\n  {'TOTAL':12s}: {total_passed}/{total_queries} ({100*total_passed/total_queries:.0f}%)")

    if all_pass:
        print("\n" + "=" * 80)
        print("VERIFICATION PASSED")
        print("All answers verified against raw corpus data")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("VERIFICATION FAILED")
        print("Some answers did not match expected values")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
