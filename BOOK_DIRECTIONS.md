# Book directions — Spam detector

**Catalog #595** · Joshua Eckroth, *Python Artificial Intelligence Projects for Beginners*, **Chapter 4** — “Detecting spam comments on YouTube”

- Packt repo: [Chapter04/SpamDetectorNN.py](https://github.com/PacktPublishing/Python-Artificial-Intelligence-Projects-for-Beginners/tree/master/Chapter04)
- Local reference: [`reference/SpamDetectorNN.py`](./reference/SpamDetectorNN.py)
- Shared logic: [`spam_core.py`](./spam_core.py)
- Labelled data (UCI YouTube Spam Collection): [`data/`](./data/) — see [`data/README.md`](./data/README.md)

## What the book asks you to build

Classify **YouTube comments** as **ham** (0) or **spam** (1) using:

1. **Keras `Tokenizer`** with the top **2,000** words and **TF-IDF** document matrices.
2. **Matrix normalization** (scale by max absolute value, subtract mean).
3. A **Sequential** network: Dense(512) → ReLU → Dropout(0.5) → Dense(2) → softmax, trained with **adamax** and **categorical crossentropy**.

Training data is the UCI CSVs (multiple artist channels); labels are in the `CLASS` column.

## Web version (`web/` — FastAPI, port **3019**)

| Book step | Web equivalent |
|-----------|----------------|
| Load CSVs + shuffle | [`spam_core.load_youtube_frames`](./spam_core.py) |
| Tokenize + TF-IDF + normalize | Training path in `spam_core` |
| Train / evaluate holdout | Background train on startup; `GET /api/status` |
| Classify new comment | `POST /api/predict` + example chips in UI |

First startup trains and caches under `web/data/cache/` (`spam_model.keras`, `tokenizer.pkl`, `meta.json`). Subsequent runs load the cache.
