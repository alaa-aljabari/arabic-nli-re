"""
nli_generator.py
----------------
Generates NLI-RE premise-hypothesis pairs from raw relation extraction records.

For each record in the raw dataset:
  - Positive (Label = True):  hypothesis from the gold relation template.
  - Negative (Label = False): hypothesis from a spurious relation template
    selected to be compatible with the domain and range of the entity pair,
    making the negative harder and more realistic.

Domain/range compatibility is enforced using RELATION_SCHEMA, which maps
each relation to its allowed subject entity types (domain) and object entity
types (range).  For no_relation records, we infer the likely entity type group
from the sentence's sentence_id context and select a spurious relation whose
domain/range matches.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

from src.data.templates import (
    POSITIVE_RELATIONS,
    RELATION_SCHEMA,
    build_nli_sentence,
    verbalize,
)


# ---------------------------------------------------------------------------
# Pre-compute: for each (domain_type, range_type) pair → compatible relations
# Used to quickly look up spurious relations given the entity type of a pair.
# ---------------------------------------------------------------------------

def _build_compatibility_index() -> Dict[Tuple[str, str], List[str]]:
    """
    Build an index: (subject_type, object_type) -> [compatible relations].

    A relation r is compatible with a (subject_type, object_type) pair if
    subject_type ∈ domain(r) AND object_type ∈ range(r).
    """
    index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for relation, (domain, range_) in RELATION_SCHEMA.items():
        for subj_type in domain:
            for obj_type in range_:
                index[(subj_type, obj_type)].append(relation)
    return dict(index)


_COMPATIBILITY_INDEX = _build_compatibility_index()

# Infer likely entity type group from relation prefix
_RELATION_TO_SUBJ_TYPE: Dict[str, str] = {
    r: list(domain)[0] if len(domain) == 1 else list(domain)[0]
    for r, (domain, _) in RELATION_SCHEMA.items()
}
_RELATION_TO_OBJ_TYPE: Dict[str, str] = {
    r: list(range_)[0] if len(range_) == 1 else list(range_)[0]
    for r, (_, range_) in RELATION_SCHEMA.items()
}


def _compatible_spurious_relations(
    gold_relation: str,
    exclude: str,
) -> List[str]:
    """
    Return relations that are domain/range compatible with gold_relation
    but are not the gold relation itself.

    Strategy: use the domain of gold_relation's subject type and the range
    of gold_relation's object type to find all relations that accept the
    same entity type combination.  This ensures the spurious hypothesis is
    plausible given the entity types in the sentence.

    Args:
        gold_relation: The actual relation of the positive triple.
        exclude:       Relation to exclude (the gold relation itself).

    Returns:
        List of compatible spurious relation keys.
    """
    if gold_relation not in RELATION_SCHEMA:
        return sorted(POSITIVE_RELATIONS - {exclude})

    subj_types, obj_types = RELATION_SCHEMA[gold_relation]

    candidates: Set[str] = set()
    for subj_type in subj_types:
        for obj_type in obj_types:
            for rel in _COMPATIBILITY_INDEX.get((subj_type, obj_type), []):
                if rel != exclude:
                    candidates.add(rel)

    # Fallback: if no compatible relation found, use all relations
    return sorted(candidates) if candidates else sorted(POSITIVE_RELATIONS - {exclude})


# ---------------------------------------------------------------------------
# Build sentence_id → gold relations index for no_relation records
# ---------------------------------------------------------------------------

def _build_sentence_gold_index(records: List[Dict]) -> Dict[str, List[str]]:
    """
    Index sentence_id → list of gold relations in that sentence.

    For no_relation records, we use this to find what relations are present
    in the same sentence, then pick a spurious relation compatible with their
    domain/range.
    """
    index: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        if rec["relation"] in POSITIVE_RELATIONS:
            index[rec["sentence_id"]].append(rec["relation"])
    return dict(index)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_nli_pairs(
    records: List[Dict],
    seed: int = 42,
) -> List[Dict]:
    """
    Convert raw RE records into binary NLI premise-hypothesis pairs,
    with domain/range-aware negative sampling.

    For positive records (known relation):
        Label = True, hypothesis = T_r(subject, object).

    For no_relation records:
        Label = False, hypothesis = T_r'(subject, object) where r' is
        selected from relations compatible with the domain and range of
        the entity pair, inferred from other gold relations in the same
        sentence.  If no gold relation exists for that sentence, fall back
        to any relation whose domain/range is broadly compatible.

    Args:
        records: List of dicts with keys: sentence_id, sentence,
                 subject, object, relation.
        seed:    Random seed for reproducibility.

    Returns:
        List of NLI instance dicts with keys: nli_sentence, Label.
    """
    random.seed(seed)
    nli_pairs: List[Dict] = []

    # Build sentence-level gold relation index for informed negative sampling
    sentence_gold_index = _build_sentence_gold_index(records)

    for rec in records:
        sentence    = rec["sentence"]
        subject     = rec["subject"]
        obj         = rec["object"]
        relation    = rec["relation"]
        sentence_id = rec.get("sentence_id", "")

        if relation in POSITIVE_RELATIONS:
            # ── Positive pair ─────────────────────────────────────────────
            hypothesis   = verbalize(relation, subject, obj)
            nli_sentence = build_nli_sentence(sentence, hypothesis)
            nli_pairs.append({
                "nli_sentence": nli_sentence,
                "Label":        "True",
            })

        elif relation == "no_relation":
            # ── Negative pair: domain/range-aware spurious relation ────────
            # 1. Find gold relations in the same sentence to infer entity types
            gold_relations = sentence_gold_index.get(sentence_id, [])

            if gold_relations:
                # Pick a random gold relation from this sentence as the
                # reference for domain/range compatibility
                ref_relation = random.choice(gold_relations)
                candidates   = _compatible_spurious_relations(
                    gold_relation=ref_relation,
                    exclude=ref_relation,
                )
            else:
                # No gold relation in this sentence — use all relations
                candidates = sorted(POSITIVE_RELATIONS)

            spurious_relation = random.choice(candidates)
            hypothesis        = verbalize(spurious_relation, subject, obj)
            nli_sentence      = build_nli_sentence(sentence, hypothesis)
            nli_pairs.append({
                "nli_sentence": nli_sentence,
                "Label":        "False",
            })

    return nli_pairs


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> List[Dict]:
    """Load records from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: List[Dict], path: str | Path) -> None:
    """Save records to a JSONL file (UTF-8, one object per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sample_balanced(
    nli_pairs: List[Dict],
    n_positive: int,
    n_negative: int,
    seed: int = 42,
) -> List[Dict]:
    """
    Draw a balanced sample of positive and negative NLI pairs.

    Args:
        nli_pairs:   Full list of generated NLI pairs.
        n_positive:  Number of True (relation present) instances to sample.
        n_negative:  Number of False (relation absent) instances to sample.
        seed:        Random seed.

    Returns:
        Shuffled list of sampled pairs.
    """
    random.seed(seed)
    positives = [p for p in nli_pairs if p["Label"] == "True"]
    negatives = [p for p in nli_pairs if p["Label"] == "False"]

    sampled = (
        random.sample(positives, min(n_positive, len(positives)))
        + random.sample(negatives, min(n_negative, len(negatives)))
    )
    random.shuffle(sampled)
    return sampled


def split_data(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split records into train / val / test sets.

    Args:
        records:     Full shuffled list of NLI pairs.
        train_ratio: Fraction for training (default 0.70).
        val_ratio:   Fraction for validation (default 0.15).
                     Test gets the remainder (1 - train - val).
        seed:        Random seed.

    Returns:
        Tuple of (train, val, test) lists.
    """
    random.seed(seed)
    data = records.copy()
    random.shuffle(data)

    n       = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]
