"""
test_dataset.py
---------------
Unit tests for ClassificationDataset.
"""

import pytest
from src.data.dataset import ClassificationDataset

MODEL_NAME = "bert-base-multilingual-cased"  # lightweight stand-in for CI
LABEL_MAP = {"entailment": 0, "contradiction": 1}
TEXTS = ["هذا اختبار.", "جملة عربية أخرى."]
TARGETS = ["entailment", "contradiction"]


@pytest.fixture()
def dataset():
    return ClassificationDataset(
        texts=TEXTS,
        targets=TARGETS,
        model_name=MODEL_NAME,
        max_len=32,
        label_map=LABEL_MAP,
    )


def test_length(dataset):
    assert len(dataset) == len(TEXTS)


def test_item_keys(dataset):
    item = dataset[0]
    assert hasattr(item, "input_ids")
    assert hasattr(item, "attention_mask")
    assert hasattr(item, "label")


def test_label_values(dataset):
    for i, target in enumerate(TARGETS):
        assert dataset[i].label == LABEL_MAP[target]


def test_padding(dataset):
    item = dataset[0]
    assert len(item.input_ids) == 32
