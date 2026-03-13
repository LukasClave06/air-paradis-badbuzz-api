import os
from dataclasses import dataclass
from typing import Dict, Any
import cloudpickle

from src.common.text_cleaning import basic_clean


@dataclass
class PredictorConfig:
    model_path: str
    threshold: float = 0.5


class SentimentPredictor:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg

        with open(cfg.model_path, "rb") as f:
            self.model = cloudpickle.load(f)

    def predict_one(self, text: str) -> Dict[str, Any]:
        cleaned = basic_clean(text)

        proba = self.model.predict_proba([cleaned])[0]
        proba_neg = float(proba[0])  # label 0 = négatif
        proba_pos = float(proba[1])  # label 1 = positif

        pred_label = 1 if proba_pos >= self.cfg.threshold else 0
        pred_text = "positif" if pred_label == 1 else "negatif"
        bad_buzz = bool(proba_neg >= self.cfg.threshold)

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