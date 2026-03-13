# Air Paradis – API de prédiction de sentiment

Ce projet fournit une API d’inférence permettant de prédire le sentiment d’un tweet
positif ou négatif pour la compagnie Air Paradis.

Le modèle utilisé est un réseau de neurones LSTM avec embeddings GloVe,
entraîné sur le dataset Sentiment140 et versionné avec MLflow.

Cette version du projet correspond au dossier de déploiement contenant :

- le modèle final
- les artifacts MLflow
- l’API Flask
- le code de preprocessing
- les tests unitaires
- les dépendances nécessaires au déploiement


------------------------------------------------------------

## Structure du projet

```
deployment/
├── mlruns/                         # Run MLflow contenant le modèle final
│   └── 249839535688655471/
│       ├── meta.yaml
│       └── 1ce1649c820d4e33ab0795e777b8cb4c/
│           ├── artifacts/
│           ├── metrics/
│           ├── params/
│           ├── tags/
│           └── meta.yaml
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                  # API Flask
│   │   └── predictor.py            # Chargement modèle + prédiction
│   │
│   └── common/
│       ├── __init__.py
│       └── text_cleaning.py        # Nettoyage du texte
│
├── tests/
│   └── test_api.py                 # Tests unitaires pytest
│
├── requirements.txt               # Dépendances runtime
├── requirements_dev.txt           # Dépendances dev / tests
├── wsgi.py                        # Lancement serveur (cloud)
├── .gitignore
└── README.md
```

------------------------------------------------------------

## Modèle retenu

Plusieurs modèles ont été testés :

- Régression logistique + TF-IDF
- Random Forest + TF-IDF
- LSTM + Word2Vec
- LSTM + GloVe
- BERT (sur échantillon réduit)

Le modèle retenu pour le déploiement est :

LSTM + embeddings GloVe

Raisons du choix :

- bonnes performances sur test
- temps d’entraînement raisonnable
- taille du modèle compatible avec le déploiement
- inférence rapide
- plus stable que BERT dans les contraintes techniques du projet

Le modèle final est versionné avec MLflow et chargé via son run_id.


------------------------------------------------------------

## Fonctionnement de l’API

L’API Flask permet :

- de charger le modèle MLflow
- de nettoyer le texte
- de tokenizer avec le tokenizer sauvegardé
- de prédire le sentiment
- de retourner les probabilités

Sortie du modèle :

- proba_pos → probabilité tweet positif
- proba_neg → probabilité tweet négatif
- pred_label → 0 ou 1
- pred_text → positif / negatif
- bad_buzz → True si proba_neg >= 0.5


------------------------------------------------------------

## Endpoints

### Vérification
```
GET /health
```
Réponse :

```
{
"status": "ok",
"run_id": "..."
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
"text": "Air Paradis is amazing"
}
```

Réponse :

```
{
"tweet": "...",
"tweet_clean": "...",
"proba_pos": 0.91,
"proba_neg": 0.09,
"pred_label": 1,
"pred_text": "positif",
"bad_buzz": false
}
```

---

### Interface web simple
```
GET /
```
Page HTML minimale permettant de tester l’API.


------------------------------------------------------------

## Prérequis

- Python 3.11
- environnement virtuel recommandé
- présence du dossier mlruns contenant le modèle


------------------------------------------------------------

## Variables d’environnement

L’API utilise les variables suivantes :
```
MLFLOW_TRACKING_URI  
```
chemin vers le dossier mlruns
```
MLFLOW_RUN_ID
```  
identifiant du run MLflow du modèle final
```
MAX_LEN
```  
longueur maximale des séquences (default = 40)
```
THRESHOLD
```  
seuil de décision (default = 0.5)


------------------------------------------------------------

## Lancement en local

Activer l’environnement virtuel :

```
venv_airparadis_train\Scripts\activate
```

Définir MLflow :

```
$env:MLFLOW_TRACKING_URI="file:///C:/chemin/vers/deployment/mlruns"
$env:MLFLOW_RUN_ID="1ce1649c820d4e33ab0795e777b8cb4c"
```

Lancer l’API :

```
python -m src.api.app
```

Ouvrir :

```
http://127.0.0.1:8000
```


------------------------------------------------------------

## Tests unitaires

Lancer :

```
pytest -v
```

Résultat attendu :


3 passed



------------------------------------------------------------

## Objectif MLOps

Ce projet démontre :

- suivi des modèles avec MLflow
- encapsulation du modèle dans une API Flask
- tests unitaires automatisés avec pytest
- structure compatible CI/CD
- séparation entraînement / déploiement
- déploiement sur plateforme Cloud


------------------------------------------------------------

## Déploiement

Le projet est conçu pour être déployé sur :

- PythonAnywhere
- serveur Flask / WSGI
- plateforme Cloud


------------------------------------------------------------

## Auteur

Lukas C  
Formation Ingénieur IA – OpenClassrooms  
Projet : Détection de bad buzz pour Air Paradis
