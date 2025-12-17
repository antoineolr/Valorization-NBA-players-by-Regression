# Projet de Valorisation des Joueurs NBA

## Description
Ce projet implémente un modèle d'évaluation de la valeur marchande des joueurs de la NBA en s'appuyant sur des principes d'économétrie et de machine learning. L'objectif est d'estimer le salaire théorique d'un athlète en fonction de ses performances statistiques et d'identifier les écarts par rapport au salaire réel.

## Fondements Théoriques
La méthodologie repose sur les travaux de Berri et al. (2004) concernant l'impact de la productivité sur les rémunérations dans le sport professionnel. Le projet utilise une approche de régression linéaire régularisée pour isoler les facteurs de performance les plus significatifs.

## Algorithme et Méthodologie

### 1. Préparation des Données
* **Filtrage statistique** : Exclusion des joueurs disposant d'un temps de jeu inférieur à un seuil défini pour garantir la fiabilité des moyennes.
* **Transformation logarithmique** : Application de la fonction $\ln$ sur la variable cible (Salaire) pour stabiliser la variance et normaliser la distribution des erreurs.
* **Encodage** : Traitement des variables catégorielles via la méthode des variables indicatrices (dummies).

### 2. Ingénierie des Caractéristiques
* **Standardisation** : Application du Z-score sur les variables explicatives afin de rendre les coefficients comparables et de permettre une régularisation Lasso efficace.

### 3. Modélisation
* **Régression Moindres Carrés Ordinaires (OLS)** : Établissement d'un modèle de référence et analyse de la significativité statistique des prédicteurs.
* **Régression Lasso** : Utilisation de la régularisation $\ell_1$ pour la sélection de variables et la gestion de la multicolinéarité.

### 4. Optimisation
* **Validation croisée (K-Fold)** : Recherche de l'hyperparamètre $\alpha$ optimal pour minimiser l'erreur quadratique moyenne ($MSE$).

### 5. Analyse des Résidus
* **Identification des écarts** : Calcul du résidu ($Salaire_{réel} - Salaire_{prédit}$).
* **Analyse de valeur** : Classification des joueurs en catégories (sur-évalués ou sous-évalués par le marché).



## Technologies
* Langage : Python
* Bibliothèques principales : Pandas, NumPy, Scikit-learn, Statsmodels
* Base de données : SQL (SQLite) via SQLAlchemy

## Structure du Répertoire
```bash
nba_valuation/
├── src/
│   ├── data_handler.py    # Acquisition et transformation
│   ├── model_trainer.py   # Entraînement et optimisation
│   └── database.py        # Persistance des résultats
├── main.py                # Point d'entrée du programme
├── requirements.txt       # Dépendances
└── README.md

