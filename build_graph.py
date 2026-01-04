#!/usr/bin/env python3
"""
Build Knowledge Graph from Structured Benchmark Data

This script builds a knowledge graph directly from the structured corpus.json
file, extracting entities and relationships from the explicit fields.

Usage:
    python3 build_graph.py
    python3 build_graph.py --input benchmark_data/corpus.json --output benchmark_data/graph_structured.json
"""

import json
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# =============================================================================
# DATA STRUCTURES (matching recursive-knowledge/scripts/graph_ops.py)
# =============================================================================

def generate_id(prefix: str, content: str) -> str:
    """Generate deterministic ID from content."""
    hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{prefix}_{hash_val}"


@dataclass
class Entity:
    id: str
    type: str
    name: str
    aliases: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    source_docs: list[str] = field(default_factory=list)
    extraction_confidence: float = 1.0


@dataclass
class Relationship:
    id: str
    type: str
    source_entity_id: str
    target_entity_id: str
    attributes: dict = field(default_factory=dict)
    source_docs: list[str] = field(default_factory=list)
    extraction_confidence: float = 1.0


@dataclass
class Document:
    id: str
    path: str
    title: str
    content_hash: str
    chunk_count: int
    indexed_at: str


class KnowledgeGraph:
    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relationships: dict[str, Relationship] = {}
        self.documents: dict[str, Document] = {}

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity

    def add_relationship(self, rel: Relationship):
        self.relationships[rel.id] = rel

    def add_document(self, doc: Document):
        self.documents[doc.id] = doc

    def save(self, path: str):
        data = {
            "entities": {k: asdict(v) for k, v in self.entities.items()},
            "relationships": {k: asdict(v) for k, v in self.relationships.items()},
            "documents": {k: asdict(v) for k, v in self.documents.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def stats(self) -> dict:
        entity_types = {}
        for e in self.entities.values():
            entity_types[e.type] = entity_types.get(e.type, 0) + 1
        return {
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "document_count": len(self.documents),
            "entity_types": entity_types,
        }


# =============================================================================
# GRAPH BUILDING
# =============================================================================

def build_graph_from_corpus(corpus_path: Path, verbose: bool = False) -> KnowledgeGraph:
    """Build knowledge graph from structured corpus data."""

    corpus = json.loads(corpus_path.read_text())
    if verbose:
        print(f"Loaded {len(corpus)} records from {corpus_path}")

    graph = KnowledgeGraph()
    entity_map = {}  # (type, name.lower()) -> entity_id

    def get_or_create_entity(name: str, etype: str, source_doc: str) -> str:
        """Get existing entity or create new one."""
        key = (etype, name.lower())
        if key in entity_map:
            ent = graph.entities[entity_map[key]]
            if source_doc not in ent.source_docs:
                ent.source_docs.append(source_doc)
            return entity_map[key]

        ent = Entity(
            id=generate_id("ent", f"{etype}:{name}"),
            type=etype,
            name=name,
            source_docs=[source_doc],
        )
        graph.add_entity(ent)
        entity_map[key] = ent.id
        return ent.id

    # Process each record
    for i, rec in enumerate(corpus):
        # Create document
        doc_id = generate_id("doc", rec['id'])
        content_hash = hashlib.sha256(rec['content'].encode()).hexdigest()[:16]
        doc = Document(
            id=doc_id,
            path=f"corpus/{rec['id']}.txt",
            title=rec['id'],
            content_hash=content_hash,
            chunk_count=1,
            indexed_at="2024-01-01",
        )
        graph.add_document(doc)

        # Extract entities
        company_id = get_or_create_entity(rec['company'], 'organization', doc_id)
        topic_id = get_or_create_entity(rec['topic'], 'concept', doc_id)
        location_id = get_or_create_entity(rec['location'], 'location', doc_id)
        category_id = get_or_create_entity(rec['category'], 'category', doc_id)

        person_ids = []
        for person in rec['people']:
            person_id = get_or_create_entity(person, 'person', doc_id)
            person_ids.append(person_id)

        # Create relationships

        # Company -> Topic (works_on)
        rel = Relationship(
            id=generate_id("rel", f"{company_id}:{topic_id}:works_on:{rec['id']}"),
            type="works_on",
            source_entity_id=company_id,
            target_entity_id=topic_id,
            attributes={"category": rec['category']},
            source_docs=[doc_id],
        )
        graph.add_relationship(rel)

        # Company -> Location (located_in)
        rel = Relationship(
            id=generate_id("rel", f"{company_id}:{location_id}:located_in:{rec['id']}"),
            type="located_in",
            source_entity_id=company_id,
            target_entity_id=location_id,
            source_docs=[doc_id],
        )
        graph.add_relationship(rel)

        # Person -> Company (works_for)
        for person_id in person_ids:
            rel = Relationship(
                id=generate_id("rel", f"{person_id}:{company_id}:works_for:{rec['id']}"),
                type="works_for",
                source_entity_id=person_id,
                target_entity_id=company_id,
                attributes={"category": rec['category']},
                source_docs=[doc_id],
            )
            graph.add_relationship(rel)

        # Person -> Topic (works_on)
        for person_id in person_ids:
            rel = Relationship(
                id=generate_id("rel", f"{person_id}:{topic_id}:works_on:{rec['id']}"),
                type="works_on",
                source_entity_id=person_id,
                target_entity_id=topic_id,
                source_docs=[doc_id],
            )
            graph.add_relationship(rel)

        if verbose and (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(corpus)} records...")

    return graph


def main():
    parser = argparse.ArgumentParser(description="Build knowledge graph from corpus")
    parser.add_argument("--input", "-i", default="benchmark_data/corpus.json",
                        help="Path to corpus.json")
    parser.add_argument("--output", "-o", default="benchmark_data/graph_structured.json",
                        help="Output path for graph")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    corpus_path = Path(args.input)
    if not corpus_path.exists():
        print(f"Error: {corpus_path} not found")
        print("Run 'python3 generate_benchmark.py' first")
        return 1

    print("Building knowledge graph...")
    graph = build_graph_from_corpus(corpus_path, args.verbose)

    output_path = args.output
    graph.save(output_path)

    print(f"\nGraph saved to {output_path}")
    print(f"Statistics: {graph.stats()}")

    return 0


if __name__ == "__main__":
    exit(main())
