"""
nli_generator.py
----------------
Generates NLI-RE premise-hypothesis pairs from raw relation extraction records.

Section 3 (§Template Construction) describes how each input sentence s with
entity pair (n_i, n_j) and relation r is converted into a binary NLI instance:

  - Positive (Label = True):  hypothesis h = T_r(n_i, n_j) for the gold relation r.
  - Negative (Label = False): hypothesis h = T_r'(n_i, n_j) for a randomly
    sampled relation r' ≠ r, or directly from no_relation records.

The output JSONL records contain all original fields plus:
  - hypothesis:    the verbalized Arabic hypothesis string h
  - nli_sentence:  the full "[CLS] s [SEP] h" string (Eq. 1)
  - Label:         "True" or "False"
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from src.data.templates import (
    POSITIVE_RELATIONS,
    RELATION_TEMPLATES,
    build_nli_sentence,
    verbalize,
)

ALL_POSITIVE_RELATIONS = sorted(POSITIVE_RELATIONS)


def generate_nli_pairs(
    records: List[Dict],
    seed: int = 42,
) -> List[Dict]:
    """
    Convert raw RE records into binary NLI premise-hypothesis pairs.

    For each record:
    - If relation is a known positive relation → Label = True, hypothesis
      from the gold template T_r(subject, object).
    - If relation is no_relation → Label = False, hypothesis from a
      randomly sampled template T_r'(subject, object) where r' ≠ no_relation.

    Args:
        records: List of dicts with keys: sentence, subject, object, relation.
        seed:    Random seed for negative sampling reproducibility.

    Returns:
        List of NLI instance dicts ready for JSONL serialisation.
    """
    random.seed(seed)
    nli_pairs: List[Dict] = []

    for rec in records:
        sentence = rec["sentence"]
        subject  = rec["subject"]
        obj      = rec["object"]
        relation = rec["relation"]

        if relation in POSITIVE_RELATIONS:
            # ── Positive pair: gold relation → True ───────────────────────
            hypothesis   = verbalize(relation, subject, obj)
            nli_sentence = build_nli_sentence(sentence, hypothesis)
            nli_pairs.append({
                "nli_sentence": nli_sentence,
                "Label":        "True",
            })

        elif relation == "no_relation":
            # ── Negative pair: random template → False ────────────────────
            rand_relation = random.choice(ALL_POSITIVE_RELATIONS)
            hypothesis    = verbalize(rand_relation, subject, obj)
            nli_sentence  = build_nli_sentence(sentence, hypothesis)
            nli_pairs.append({
                "nli_sentence": nli_sentence,
                "Label":        "False",
            })

    return nli_pairs


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
