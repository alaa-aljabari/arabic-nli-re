# NLI-RE: Arabic Relation Extraction as Natural Language Inference

Implementation of **NLI-RE**, a framework that reformulates relation extraction (RE) as a Natural Language Inference (NLI) task. Given an input sentence *s* containing two named entity mentions *n_i* and *n_j*, NLI-RE treats *s* as the premise and automatically constructs a hypothesis by verbalizing a candidate relation *r* between *n_i* and *n_j* using a predefined Arabic template. A transformer encoder then predicts whether the premise **entails** the hypothesis — indicating the presence of relation *r* between the entity pair.

---

## Table of Contents

- [Framework Overview](#framework-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Usage](#usage)
- [License](#license)

---

## Framework Overview

NLI-RE proceeds through three steps (Section 3):

### 1. Template Construction
For each of the **40 relation types**, a relation-aware Arabic template `T_r(e1, e2)` verbalizes a candidate relation between subject entity `e1` and object entity `e2` into a hypothesis string `h`. For example:

| Relation | Subject | Object | Hypothesis h |
|---|---|---|---|
| `Location.located_in` | قبة الصخرة | مدينة القدس | قبة الصخرة يقع في مدينة القدس |
| `Personal.has_occupation` | حبيب بولس | مدير مال عكا | حبيب بولس يعمل مدير مال عكا |
| `Affiliation.member_of` | أطباء بلا حدود | الأمم المتحدة | أطباء بلا حدود عضو في الأمم المتحدة |

### 2. Sentence Encoder (Eq. 1)
A transformer encoder `T` (UBC-NLP/ARBERTv2) processes the concatenated premise-hypothesis pair:

```
H = T([CLS] s [SEP] h)
```

### 3. Relation Inference (Eq. 2)
The feature vector `H` is passed through a fully connected layer:

```
ŷ_i = σ(H W + b)
```

**True** → relation `r` exists between `(n_i, n_j)` in `s`.  
**False** → relation `r` does not exist.

### Training Objective

```
Loss = L_WCE + L_NCE
```

**L_WCE** (Eq. 3) — Weighted Cross-Entropy with `[w_n, w_p]` to address class imbalance.  
**L_NCE** (Eq. 4) — Noise Contrastive Estimation with temperature `τ` to improve discrimination.

---

## Project Structure

```
arabic-nli-classifier/
│
├── configs/
│   └── config.yaml                  # All hyperparameters and paths
│
├── data/
│   ├── nli/                         # ✅ Committed — NLI sample pairs (train/val/test)
│   │   ├── train.jsonl              #    280 premise-hypothesis pairs
│   │   ├── val.jsonl                #     60 pairs
│   │   └── test.jsonl               #     60 pairs
│   └── raw/                         # ❌ Not committed — place full dataset here
│       └── train.jsonl              #    Full RE records (sentence/subject/object/relation)
│
├── src/
│   ├── data/
│   │   ├── templates.py             # 40 Arabic relation verbalization templates T_r
│   │   ├── nli_generator.py         # Generates [CLS] s [SEP] h pairs from RE records
│   │   ├── dataset.py               # PyTorch Dataset — Sentence Encoder input (Eq. 1)
│   │   └── loader.py                # Loads NLI JSONL splits into DataFrames
│   │
│   ├── models/
│   │   └── trainer.py               # ContrastiveTrainer: Loss = L_WCE + L_NCE
│   │
│   ├── training/
│   │   └── cross_val.py             # Stratified K-Fold fine-tuning loop
│   │
│   ├── evaluation/
│   │   └── metrics.py               # Micro-F1, ensemble inference, JSONL reporting
│   │
│   └── utils/
│       └── helpers.py               # Seed control, label maps, config loader
│
├── scripts/
│   ├── prepare_data.py              # Step 0: generate NLI pairs from raw RE records
│   ├── train.py                     # Step 1: K-Fold training
│   └── predict.py                   # Step 2: ensemble inference
│
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_loss.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/aaljabari/arabic-nli-re.git
cd arabic-nli-re

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Dataset

### Included sample (ready to use)

The repository includes a balanced sample of **400 NLI pairs** in `data/nli/`:

| File | Pairs | True | False |
|---|---|---|---|
| `train.jsonl` | 280 | ~140 | ~140 |
| `val.jsonl`   |  60 | ~30  | ~30  |
| `test.jsonl`  |  60 | ~30  | ~30  |

Each record has the structure:

```json
{
  "sentence_id":  "100007145",
  "sentence":     "رسالة من مدير مال عكا حبيب بولس إلى وكيل قاضي عكا ...",
  "subject":      "حبيب بولس",
  "object":       "مدير مال عكا",
  "relation":     "Personal.has_occupation",
  "hypothesis":   "حبيب بولس يعمل مدير مال عكا",
  "nli_sentence": "[CLS] رسالة من مدير مال عكا ... [SEP] حبيب بولس يعمل مدير مال عكا",
  "Label":        "True"
}
```

### Using your own full dataset

Place your raw RE records at `data/raw/train.jsonl`, then run:

```bash
python scripts/prepare_data.py --config configs/config.yaml

# Custom sample size:
python scripts/prepare_data.py --n_positive 500 --n_negative 500
```

Raw records must have the fields: `sentence`, `subject`, `object`, `relation`.

---

## Configuration

```yaml
model:
  name: "UBC-NLP/ARBERTv2"   # Sentence encoder T
  max_len: 128                 # Max tokens for [CLS] s [SEP] h
  num_folds: 2                 # K in Stratified K-Fold

loss:
  tau: 1.0                     # Temperature τ — L_NCE (Eq. 4)
  class_weights: [0.2, 1.0]   # [w_n, w_p]  — L_WCE (Eq. 3)

data:
  train_path: "data/nli/train.jsonl"
  dev_path:   "data/nli/val.jsonl"
  test_path:  "data/nli/test.jsonl"
```

---

## Usage

### Run on the included sample (no setup needed)

```bash
# Train
python scripts/train.py --config configs/config.yaml

# Predict
python scripts/predict.py --config configs/config.yaml
```

### Run on the full dataset

```bash
# Step 0 — generate NLI pairs
python scripts/prepare_data.py --config configs/config.yaml

# Step 1 — train
python scripts/train.py --config configs/config.yaml

# Step 2 — predict
python scripts/predict.py --config configs/config.yaml
```

---

## License

[MIT](LICENSE)
