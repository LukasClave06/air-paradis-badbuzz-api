import json
from dataclasses import dataclass
from typing import Dict, Any
import mlflow
import numpy as np
from mlflow.artifacts import download_artifacts
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from src.common.text_cleaning import basic_clean


@dataclass
class PredictorConfig:
    run_id: str
    max_len: int = 40
    threshold: float = 0.5  # seuil côté "positif" (proba_pos)


class SentimentPredictor:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg

        self.model_uri = f"runs:/{cfg.run_id}/model"
        self.tok_uri = f"runs:/{cfg.run_id}/lstm_glove_tokenizer.json"

        # 1) Load TF model from MLflow
        self.model = mlflow.tensorflow.load_model(self.model_uri)

        # 2) Load tokenizer JSON artifact from MLflow
        tok_path = download_artifacts(self.tok_uri)
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_json = f.read()
        self.tokenizer = tokenizer_from_json(tok_json)

    def predict_one(self, text: str) -> Dict[str, Any]:
        # same cleaning as training
        cleaned = basic_clean(text)

        # tokenize + pad
        seq = self.tokenizer.texts_to_sequences([cleaned])
        X = pad_sequences(seq, maxlen=self.cfg.max_len, padding="post", truncating="post")

        # model output: sigmoid -> proba positive (label=1)
        proba_pos = float(self.model.predict(X, verbose=0).ravel()[0])
        proba_neg = float(1.0 - proba_pos)

        # Decision:
        pred_label = 1 if proba_pos >= self.cfg.threshold else 0
        pred_text = "positif" if pred_label == 1 else "negatif"

        # "Bad buzz" business flag (négatif prioritaire)
        bad_buzz = bool(proba_neg >= self.cfg.threshold)  # équivalent au seuil sur proba_pos

        return {
            "tweet": text,
            "tweet_clean": cleaned,

            "proba_pos": proba_pos,
            "proba_neg": proba_neg,

            "pred_label": pred_label,
            "pred_text": pred_text,

            "threshold": self.cfg.threshold,
            "bad_buzz": bad_buzz,
        }