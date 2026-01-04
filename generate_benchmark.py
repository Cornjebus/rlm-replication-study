#!/usr/bin/env python3
"""
Synthetic Benchmark Generator

Creates test data that mirrors OOLONG's structure:
- Large context (100K+ tokens simulated via many small records)
- Queries requiring counting, classification, aggregation
- Multi-hop reasoning needed

This lets you test without needing HuggingFace access.

Usage:
    python3 generate_benchmark.py --records 1000 --queries 20
    
Output:
    benchmark_data/
        corpus.json       # The documents/records
        queries.json      # Questions with ground truth answers
        
Then run:
    python3 index_corpus.py --input benchmark_data/corpus.json --output benchmark_data/graph.json
    python3 run_benchmark.py --graph benchmark_data/graph.json --queries benchmark_data/queries.json
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
import argparse


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

# Entity pools for generating realistic records
COMPANIES = [
    "Anthropic", "OpenAI", "Google", "Microsoft", "Meta", "Amazon", "Apple",
    "DeepMind", "Cohere", "Mistral", "Stability AI", "Hugging Face", "Nvidia",
    "Tesla", "SpaceX", "Palantir", "Databricks", "Snowflake", "MongoDB",
    "Stripe", "Square", "Coinbase", "Robinhood", "Plaid", "Affirm"
]

PEOPLE = [
    "Dario Amodei", "Sam Altman", "Demis Hassabis", "Yann LeCun", "Geoffrey Hinton",
    "Ilya Sutskever", "Andrej Karpathy", "Fei-Fei Li", "Andrew Ng", "Yoshua Bengio",
    "Ian Goodfellow", "Chris Manning", "Percy Liang", "Sergey Levine", "Chelsea Finn",
    "Pieter Abbeel", "Stuart Russell", "Max Tegmark", "Nick Bostrom", "Eliezer Yudkowsky"
]

TOPICS = [
    "machine learning", "neural networks", "transformers", "reinforcement learning",
    "computer vision", "natural language processing", "robotics", "AI safety",
    "large language models", "diffusion models", "multimodal AI", "AI alignment",
    "autonomous systems", "recommendation systems", "speech recognition"
]

LOCATIONS = [
    "San Francisco", "New York", "London", "Toronto", "Montreal", "Seattle",
    "Boston", "Austin", "Beijing", "Shanghai", "Tokyo", "Singapore", "Berlin",
    "Paris", "Tel Aviv", "Bangalore", "Sydney"
]

CATEGORIES = ["research", "product", "funding", "partnership", "acquisition", "hiring", "policy"]


@dataclass
class Record:
    """A single record in the corpus."""
    id: str
    category: str
    company: str
    people: List[str]
    topic: str
    location: str
    date: str
    content: str


def generate_record(record_id: int) -> Record:
    """Generate a single synthetic record."""
    category = random.choice(CATEGORIES)
    company = random.choice(COMPANIES)
    people = random.sample(PEOPLE, k=random.randint(1, 3))
    topic = random.choice(TOPICS)
    location = random.choice(LOCATIONS)
    
    year = random.randint(2020, 2025)
    month = random.randint(1, 12)
    date = f"{year}-{month:02d}"
    
    # Generate content based on category
    if category == "research":
        content = f"{company} researchers including {', '.join(people)} published new work on {topic}. The research, conducted at their {location} office, introduces novel approaches to {random.choice(TOPICS)}."
    elif category == "product":
        content = f"{company} announced a new {topic} product. {people[0]} led the development team based in {location}. The product aims to improve {random.choice(TOPICS)} capabilities."
    elif category == "funding":
        amount = random.randint(10, 500) * 10
        content = f"{company} raised ${amount}M in funding for their {topic} initiatives. {people[0]} commented on the investment from their {location} headquarters."
    elif category == "partnership":
        partner = random.choice([c for c in COMPANIES if c != company])
        content = f"{company} partnered with {partner} on {topic}. {people[0]} and team in {location} will lead the collaboration on {random.choice(TOPICS)}."
    elif category == "acquisition":
        target = f"{random.choice(['AI', 'ML', 'Tech'])}Startup"
        content = f"{company} acquired {target}, a {topic} company. {people[0]} announced the deal will expand their {location} operations."
    elif category == "hiring":
        content = f"{company} hired {people[0]} to lead {topic} efforts. The new role is based in {location} and focuses on {random.choice(TOPICS)}."
    else:  # policy
        content = f"{company}'s {people[0]} testified on {topic} policy. The {location}-based executive discussed implications for {random.choice(TOPICS)}."
    
    return Record(
        id=f"rec_{record_id:05d}",
        category=category,
        company=company,
        people=people,
        topic=topic,
        location=location,
        date=date,
        content=content,
    )


def generate_corpus(n_records: int) -> List[Record]:
    """Generate full corpus."""
    return [generate_record(i) for i in range(n_records)]


# =============================================================================
# QUERY GENERATION (OOLONG-style)
# =============================================================================

@dataclass
class Query:
    """A benchmark query with ground truth."""
    id: str
    question: str
    query_type: str  # count, classify, aggregate, multi_hop
    answer: str
    reasoning_steps: List[str]


def generate_counting_query(corpus: List[Record], query_id: int) -> Query:
    """Generate a counting query (OOLONG trec_coarse style)."""
    # Pick a filter criterion
    filter_type = random.choice(["company", "category", "location", "topic"])
    
    if filter_type == "company":
        company = random.choice(COMPANIES)
        matching = [r for r in corpus if r.company == company]
        question = f"How many records mention {company}?"
        answer = str(len(matching))
        reasoning = [f"Filter records where company == '{company}'", f"Count: {len(matching)}"]
    
    elif filter_type == "category":
        category = random.choice(CATEGORIES)
        matching = [r for r in corpus if r.category == category]
        question = f"How many {category} records are there?"
        answer = str(len(matching))
        reasoning = [f"Filter records where category == '{category}'", f"Count: {len(matching)}"]
    
    elif filter_type == "location":
        location = random.choice(LOCATIONS)
        matching = [r for r in corpus if r.location == location]
        question = f"How many records are from {location}?"
        answer = str(len(matching))
        reasoning = [f"Filter records where location == '{location}'", f"Count: {len(matching)}"]
    
    else:  # topic
        topic = random.choice(TOPICS)
        matching = [r for r in corpus if r.topic == topic]
        question = f"How many records discuss {topic}?"
        answer = str(len(matching))
        reasoning = [f"Filter records where topic == '{topic}'", f"Count: {len(matching)}"]
    
    return Query(
        id=f"q_{query_id:03d}",
        question=question,
        query_type="count",
        answer=answer,
        reasoning_steps=reasoning,
    )


def generate_multi_hop_query(corpus: List[Record], query_id: int) -> Query:
    """Generate multi-hop query requiring connection across records."""
    # Find a person who appears in multiple records
    person_counts = {}
    for r in corpus:
        for p in r.people:
            person_counts[p] = person_counts.get(p, 0) + 1
    
    # Pick someone with multiple records
    multi_record_people = [p for p, c in person_counts.items() if c >= 2]
    if not multi_record_people:
        # Fallback to counting
        return generate_counting_query(corpus, query_id)
    
    person = random.choice(multi_record_people)
    person_records = [r for r in corpus if person in r.people]
    
    # Query: What companies has X worked with?
    companies = list(set(r.company for r in person_records))
    question = f"What companies has {person} been involved with?"
    answer = ", ".join(sorted(companies))
    reasoning = [
        f"Find all records mentioning {person}",
        f"Found {len(person_records)} records",
        f"Extract unique companies: {companies}",
    ]
    
    return Query(
        id=f"q_{query_id:03d}",
        question=question,
        query_type="multi_hop",
        answer=answer,
        reasoning_steps=reasoning,
    )


def generate_aggregation_query(corpus: List[Record], query_id: int) -> Query:
    """Generate aggregation query."""
    # Which company has the most records of type X?
    category = random.choice(CATEGORIES)
    
    company_counts = {}
    for r in corpus:
        if r.category == category:
            company_counts[r.company] = company_counts.get(r.company, 0) + 1
    
    if not company_counts:
        return generate_counting_query(corpus, query_id)
    
    top_company = max(company_counts, key=company_counts.get)
    question = f"Which company has the most {category} records?"
    answer = top_company
    reasoning = [
        f"Filter to category == '{category}'",
        f"Group by company and count",
        f"Top company: {top_company} with {company_counts[top_company]} records",
    ]
    
    return Query(
        id=f"q_{query_id:03d}",
        question=question,
        query_type="aggregate",
        answer=answer,
        reasoning_steps=reasoning,
    )


def generate_queries(corpus: List[Record], n_queries: int) -> List[Query]:
    """Generate mix of query types."""
    queries = []
    
    for i in range(n_queries):
        query_type = random.choice(["count", "count", "multi_hop", "aggregate"])
        
        if query_type == "count":
            q = generate_counting_query(corpus, i)
        elif query_type == "multi_hop":
            q = generate_multi_hop_query(corpus, i)
        else:
            q = generate_aggregation_query(corpus, i)
        
        queries.append(q)
    
    return queries


# =============================================================================
# OUTPUT
# =============================================================================

def save_benchmark(corpus: List[Record], queries: List[Query], output_dir: Path):
    """Save benchmark data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save corpus as individual text files (for indexing)
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    
    for record in corpus:
        record_file = corpus_dir / f"{record.id}.txt"
        record_text = f"""Record ID: {record.id}
Category: {record.category}
Company: {record.company}
People: {', '.join(record.people)}
Topic: {record.topic}
Location: {record.location}
Date: {record.date}

{record.content}
"""
        record_file.write_text(record_text)
    
    # Save corpus metadata
    corpus_meta = output_dir / "corpus.json"
    corpus_meta.write_text(json.dumps([asdict(r) for r in corpus], indent=2))
    
    # Save queries
    queries_file = output_dir / "queries.json"
    queries_file.write_text(json.dumps([asdict(q) for q in queries], indent=2))
    
    print(f"Saved {len(corpus)} records to {corpus_dir}")
    print(f"Saved {len(queries)} queries to {queries_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark")
    parser.add_argument("--records", type=int, default=1000, help="Number of records")
    parser.add_argument("--queries", type=int, default=20, help="Number of queries")
    parser.add_argument("--output", type=str, default="./benchmark_data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    print(f"Generating {args.records} records...")
    corpus = generate_corpus(args.records)
    
    print(f"Generating {args.queries} queries...")
    queries = generate_queries(corpus, args.queries)
    
    output_dir = Path(args.output)
    save_benchmark(corpus, queries, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK GENERATED")
    print("=" * 60)
    print(f"Records: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"  - Counting: {sum(1 for q in queries if q.query_type == 'count')}")
    print(f"  - Multi-hop: {sum(1 for q in queries if q.query_type == 'multi_hop')}")
    print(f"  - Aggregation: {sum(1 for q in queries if q.query_type == 'aggregate')}")
    
    # Estimate context size
    total_chars = sum(len(r.content) for r in corpus)
    print(f"\nTotal corpus size: {total_chars:,} chars (~{total_chars//4:,} tokens)")
    
    print(f"\nNext steps:")
    print(f"  1. Index corpus:")
    print(f"     python3 recursive-knowledge/scripts/index_corpus.py \\")
    print(f"       --input {output_dir}/corpus --output {output_dir}/graph.json -v")
    print(f"")
    print(f"  2. Run queries:")
    print(f"     python3 run_benchmark.py --graph {output_dir}/graph.json \\")
    print(f"       --queries {output_dir}/queries.json")


if __name__ == "__main__":
    main()
