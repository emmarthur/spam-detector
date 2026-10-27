"""
YouTube comment spam detector — TF-IDF tokenization + dense neural network.

Follows Joshua Eckroth (Packt Ch. 4, SpamDetectorNN.py): Tokenizer (2000 words),
TF-IDF matrix normalization, Sequential Dense network, ham vs spam (CLASS 0/1).
"""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

NUM_WORDS = 2000
EPOCHS = 10
BATCH_SIZE = 16
RANDOM_SEED = 42
CV_FOLDS = 5

CSV_FILES = (
    "Youtube01-Psy.csv",
    "Youtube02-KatyPerry.csv",
    "Youtube03-LMFAO.csv",
    "Youtube04-Eminem.csv",
    "Youtube05-Shakira.csv",
)


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def default_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "web" / "data" / "cache"


def load_youtube_frames(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or default_data_dir()
    frames = []
    for name in CSV_FILES:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        frames.append(pd.read_csv(path))
    combined = pd.concat(frames, ignore_index=True)
    return combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    max_abs = np.amax(np.absolute(matrix))
    if max_abs > 0:
        matrix = matrix / max_abs
    matrix = matrix - np.mean(matrix)
    return matrix


def _build_model() -> Sequential:
    model = Sequential()
    model.add(Dense(512, input_shape=(NUM_WORDS,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adamax",
        metrics=["accuracy"],
    )
    return model


def _train_on_indices(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[Sequential, Tokenizer, float]:
    train_content = frame["CONTENT"].iloc[train_idx]
    test_content = frame["CONTENT"].iloc[test_idx]

    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(train_content)

    train_inputs = _normalize_matrix(
        tokenizer.texts_to_matrix(train_content, mode="tfidf")
    )
    test_inputs = _normalize_matrix(
        tokenizer.texts_to_matrix(test_content, mode="tfidf")
    )

    train_outputs = to_categorical(frame["CLASS"].iloc[train_idx])
    test_outputs = to_categorical(frame["CLASS"].iloc[test_idx])

    model = _build_model()
    model.fit(
        train_inputs,
        train_outputs,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    scores = model.evaluate(test_inputs, test_outputs, verbose=0)
    accuracy = float(scores[1])
    return model, tokenizer, accuracy


@dataclass
class SpamPipeline:
    model: Sequential
    tokenizer: Tokenizer
    holdout_accuracy: float
    cv_mean_accuracy: float | None

    def predict(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Enter a comment to classify.")

        matrix = self.tokenizer.texts_to_matrix([cleaned], mode="tfidf")
        matrix = _normalize_matrix(matrix)
        probs = self.model.predict(matrix, verbose=0)[0]
        spam_prob = float(probs[1])
        ham_prob = float(probs[0])
        label = 1 if spam_prob >= ham_prob else 0

        return {
            "label": label,
            "classification": "spam" if label == 1 else "ham",
            "confidence": round(max(spam_prob, ham_prob) * 100, 1),
            "probabilities": {
                "ham": round(ham_prob * 100, 1),
                "spam": round(spam_prob * 100, 1),
            },
        }


def train_pipeline(
    data_dir: Path | None = None,
    *,
    compute_cv: bool = False,
) -> SpamPipeline:
    frame = load_youtube_frames(data_dir)
    labels = frame["CLASS"].to_numpy()

    train_idx, test_idx = train_test_split(
        np.arange(len(frame)),
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    model, tokenizer, holdout = _train_on_indices(frame, train_idx, test_idx)

    cv_mean: float | None = None
    if compute_cv:
        kfold = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        cv_scores: list[float] = []
        for fold_train, fold_test in kfold.split(frame, labels):
            _, _, acc = _train_on_indices(frame, fold_train, fold_test)
            cv_scores.append(acc * 100)
        cv_mean = round(float(np.mean(cv_scores)), 2)

    return SpamPipeline(
        model=model,
        tokenizer=tokenizer,
        holdout_accuracy=round(holdout * 100, 2),
        cv_mean_accuracy=cv_mean,
    )


def save_pipeline(pipeline: SpamPipeline, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline.model.save(cache_dir / "spam_model.keras")
    with (cache_dir / "tokenizer.pkl").open("wb") as handle:
        pickle.dump(pipeline.tokenizer, handle)
    meta = {
        "holdout_accuracy": pipeline.holdout_accuracy,
        "cv_mean_accuracy": pipeline.cv_mean_accuracy,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_pipeline(cache_dir: Path) -> SpamPipeline | None:
    model_path = cache_dir / "spam_model.keras"
    tokenizer_path = cache_dir / "tokenizer.pkl"
    meta_path = cache_dir / "meta.json"
    if not model_path.is_file() or not tokenizer_path.is_file():
        return None
    model = load_model(model_path)
    with tokenizer_path.open("rb") as handle:
        tokenizer = pickle.load(handle)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return SpamPipeline(
        model=model,
        tokenizer=tokenizer,
        holdout_accuracy=meta.get("holdout_accuracy", 0.0),
        cv_mean_accuracy=meta.get("cv_mean_accuracy"),
    )


def get_or_train_pipeline(
    cache_dir: Path | None = None,
    data_dir: Path | None = None,
    *,
    force_retrain: bool = False,
    compute_cv: bool = False,
) -> SpamPipeline:
    cache_dir = cache_dir or default_cache_dir()
    if not force_retrain:
        loaded = load_pipeline(cache_dir)
        if loaded is not None:
            return loaded
    pipeline = train_pipeline(data_dir=data_dir, compute_cv=compute_cv)
    save_pipeline(pipeline, cache_dir)
    return pipeline


def sample_comments() -> list[dict[str, str]]:
    return [
        {"text": "Great video, thanks for sharing!", "hint": "ham"},
        {"text": "Subscribe to my channel for free iPhones!!!", "hint": "spam"},
        {"text": "I only came here to check the view count.", "hint": "ham"},
    ]
