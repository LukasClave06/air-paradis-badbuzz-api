# Air Paradis – API de prédiction de sentiment (Bad Buzz Detection)

## Objectif du projet

Ce projet a pour objectif de détecter automatiquement le sentiment de tweets mentionnant la compagnie fictive **Air Paradis**, afin d’anticiper les situations de **bad buzz**.

Deux approches ont été développées :

- Modèle avancé (Deep Learning) : LSTM + embeddings GloVe
- Modèle classique : TF-IDF + Logistic Regression

Le modèle avancé offre de meilleures performances, mais le modèle classique a été retenu pour le déploiement afin de respecter les contraintes techniques des offres cloud gratuites.

---

## Choix du modèle déployé

| Modèle | Performance | Taille | Déploiement cloud gratuit |
|--------|-----------|---------|----------------------------|
| LSTM + GloVe | ⭐⭐⭐⭐ | Très lourd | ❌ trop volumineux |
| BERT | ⭐⭐⭐⭐⭐ | Très lourd | ❌ impossible en free tier |
| LogReg + TF-IDF | ⭐⭐⭐ | Léger | ✅ compatible |

Le modèle déployé dans l’API est donc :

> Logistic Regression + TF-IDF enregistré avec MLflow

Le modèle est chargé via MLflow à partir d’un run local.

---

## Architecture du projet

```
deployment/
├── .github/
│ └── workflows/
│ └── ci.yml
│
├── mlruns/
│ └── 249839535688655471/
│ └── be260e8b8f3e4ae7b3018afa00a828ec/
│ └── artifacts/
│ └── model/
│
├── src/
│ ├── api/
│ │ ├── init.py
│ │ ├── app.py
│ │ └── predictor.py
│ │
│ └── common/
│ ├── init.py
│ └── text_cleaning.py
│
├── tests/
│ └── test_api.py
│
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
└── wsgi.py
```

---

## Fonctionnement de l’API

L’API Flask expose :

### Health check


GET /health


Retour :

```
{
"status": "ok",
"model_type": "logreg_tfidf"
}
```

---

### Prédiction

```
POST /predict
```

Body :

```
{
"text": "Air Paradis is amazing!"
}
```

Réponse :

```
{
"tweet": "...",
"tweet_clean": "...",
"proba_pos": 0.93,
"proba_neg": 0.07,
"pred_label": 1,
"pred_text": "positif",
"bad_buzz": false
}
```

---

### Interface web

```
GET /
```

Permet de tester la prédiction via un formulaire HTML.

---

## MLflow

Le modèle est chargé depuis MLflow :

```
runs:/be260e8b8f3e4ae7b3018afa00a828ec/model
```

Tracking local :

```
deployment/mlruns/
```

Variables utilisées :

```
MLFLOW_TRACKING_URI
MLFLOW_RUN_ID
THRESHOLD
```

---

## Tests unitaires

Tests réalisés avec pytest.

Lancer :

```
pytest -v
```

Configuration :

```
pytest.ini
```

Permet d’importer correctement le package src.

---

## CI/CD avec GitHub Actions

Workflow :

```
.github/workflows/ci.yml
```

À chaque push :

- installation des dépendances
- lancement pytest
- validation automatique

Objectif :

> garantir que l’API fonctionne avant déploiement

---

## Déploiement cloud

Déploiement prévu sur :

- PythonAnywhere (free tier)
- Azure (optionnel)
- autres plateformes compatibles WSGI

Fichier utilisé :

```
wsgi.py
```

Le modèle chargé est le modèle MLflow local.

---

## Lancer en local

Activer l’environnement :

```
venv\Scripts\activate
```

Définir variables :

```
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_RUN_ID=be260e8b8f3e4ae7b3018afa00a828ec
```

Lancer :

```
python -m src.api.app
```

Puis :

```
http://127.0.0.1:8000
```

---

## Auteur

Projet réalisé dans le cadre de la formation IA / Data Scientist.

Objectifs pédagogiques :

- MLflow
- API Flask
- CI/CD
- GitHub Actions
- Déploiement cloud
- MLOps
