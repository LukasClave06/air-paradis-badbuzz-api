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


## Structure du projet

deployment/
│
├── mlruns/ # Run MLflow contenant le modèle final
├── src/
│ ├── api/
│ │ ├── app.py # API Flask
│ │ └── predictor.py # Chargement modèle + prédiction
│ │
│ └── common/
│ └── text_cleaning.py # Nettoyage du texte
│
├── tests/ # Tests unitaires pytest
├── requirements.txt # Dépendances runtime
├── requirements_dev.txt # Dépendances dev / tests
├── wsgi.py # Lancement serveur (cloud)
└── README.md



## Modèle utilisé

Modèle final :

- Architecture : LSTM + embeddings GloVe
- Dataset : Sentiment140
- Suivi des expérimentations : MLflow
- Chargement du modèle via run_id

Sortie du modèle :

- proba_pos → probabilité tweet positif
- proba_neg → probabilité tweet négatif
- pred_label → 0 ou 1
- pred_text → positif / negatif
- bad_buzz → True si proba_neg >= 0.5


## API Flask

L’API expose 3 endpoints.


### Vérification de l’API

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


### Interface web simple

```
GET /
```

Page HTML minimale permettant de tester l’API.


## Lancement en local

Activer l’environnement virtuel :

```
venv_airparadis_train\Scripts\activate
```

Définir les variables MLflow :

```
$env:MLFLOW_TRACKING_URI="file:///C:/chemin/vers/deployment/mlruns"
$env:MLFLOW_RUN_ID="RUN_ID"
```

Lancer l’API :

```
python -m src.api.app
```

Ouvrir :

```
http://127.0.0.1:8000
```


## Tests unitaires

Lancer :

```
pytest -v
```

Résultat attendu :

```
3 passed
```


## Objectif MLOps

Ce projet démontre :

- suivi des modèles avec MLflow
- encapsulation du modèle dans une API Flask
- tests unitaires avec pytest
- structure compatible CI/CD
- déploiement sur plateforme Cloud (PythonAnywhere)


## Auteur

Lukas C  
Formation Ingénieur IA – OpenClassrooms  
Projet : Détection de bad buzz pour Air Paradis
```

