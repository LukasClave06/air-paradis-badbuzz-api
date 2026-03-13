import os
from dataclasses import dataclass
from typing import Dict, Any

import mlflow
import mlflow.sklearn

from src.common.text_cleaning import basic_clean


@dataclass
class PredictorConfig:
    run_id: str
    threshold: float = 0.5


class SentimentPredictor:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.model_uri = f"runs:/{cfg.run_id}/model"

        # Pipeline sklearn complet : TF-IDF + LogisticRegression
        self.model = mlflow.sklearn.load_model(self.model_uri)

    def predict_one(self, text: str) -> Dict[str, Any]:
        cleaned = basic_clean(text)

        # Le pipeline sklearn attend du texte brut
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