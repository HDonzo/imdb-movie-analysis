IMDb Movie Analysis

🎯 Objectif

Ce projet analyse le dataset public IMDb afin de :

Explorer la distribution des films et séries selon leurs genres.

Étudier l’évolution de la production dans le temps.

Identifier les genres dominants et leur popularité.

Comprendre la relation entre notes (averageRating) et nombre de votes (numVotes).

Mettre en place un modèle de prédiction des notes.


🗂️ Données utilisées

name.basics.tsv → Infos sur les personnes (acteurs, réalisateurs, etc.)

title.akas.tsv → Titres alternatifs selon pays/langues

title.basics.tsv → Métadonnées (titre, type, années, genres)

title.principals.tsv → Casting principal

title.ratings.tsv → Notes et nombre de votes


⚙️ Méthodologie

1. Nettoyage des données (Pandas)

Suppression des doublons et valeurs manquantes.

Conversion en numérique (startYear, numVotes, averageRating).

Gestion des genres multi-labels avec MultiLabelBinarizer.

2. Visualisations (Matplotlib / Seaborn)

Distribution des genres (Top 10).

Comparaison temporelle (2000–2010 vs 2011–2025).

Scatterplots : numVotes vs averageRating.

3. Modélisation (Scikit-Learn)

Régression linéaire.

Random Forest Regressor.

Évaluation : RMSE et R².


📊 Résultats clés

🔝 Top 10 Genres les plus fréquents

➡️ Les genres dominants : Drama, Comedy, Documentary, Action.

📆 Évolution temporelle

➡️ Explosion du nombre de films et séries produits après 2010.

⭐ Notes et Votes

➡️ Corrélation claire : plus un film reçoit de votes, plus sa note est représentative.


🤖 Modélisation

Régression linéaire : performances limitées (ne capte pas les non-linéarités).

Random Forest : bien plus robuste, meilleur R² et RMSE plus bas.


📂 Organisation du projet

IMDB/
│── data/                # Fichiers TSV IMDb
│── notebooks/           # Notebooks exploratoires
│── outputs/             # Graphiques générés (.png)
│── src/                 # Scripts Python
│── project.ipynb        # Notebook principal (exploration + visualisation)
│── predict.ipynb        # Notebook de modélisation
│── README.md            # Présentation du projet
│── requirements.txt     # Dépendances Python


🚀 Installation & Utilisation

Cloner le repo :

git clone https://github.com/HDonzo/imdb-movie-analysis.git
cd imdb-movie-analysis


Installer les dépendances :


pip install -r requirements.txt


Lancer Jupyter Notebook :


jupyter notebook


Exécuter :


project.ipynb → Exploration et visualisations.

predict.ipynb → Modélisation et prédictions.


🔮 Améliorations futures

Ajouter un modèle deep learning (TensorFlow/PyTorch).

Construire un dashboard interactif (Streamlit / Plotly).

Étudier la popularité par pays/langue via title.akas.tsv.

Faire un clustering de films par similarité (genres, votes, durée).



👨‍💻 Auteur

Holo Amon-Donovan (@HDonzo)

Projet académique / exploratoire sur les données IMDb.
