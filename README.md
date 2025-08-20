# 🎬 Analyse et Prédiction sur le Dataset IMDB

## 📌 Contexte du projet

Ce projet a pour objectif d’explorer et de modéliser un large jeu de données issu d’IMDB (`title.basics.tsv` et `title.ratings.tsv`).  
L’étude se décompose en deux grandes parties :  

1. **Exploration et nettoyage des données** (Notebook `project.ipynb`)  
2. **Prédiction de la note moyenne (`averageRating`)** à partir des métadonnées (Notebooks `predict.ipynb` et `Prediction2.ipynb`).  

Ce projet a également été l’occasion de se confronter à de **nombreux problèmes pratiques** (gestion mémoire, erreurs pandas/sklearn, soucis git…), et de mettre en place des solutions robustes.


---

## 🗂 Organisation du projet

```bash
imdb-movie-analysis/
│
├── data/
│   └── raw/                     # Données brutes IMDB (.tsv)
│
├── notebooks/
│   ├── project.ipynb            # Phase exploration + nettoyage
│   ├── predict.ipynb            # 1ère tentative de modélisation (RAM limitée)
│   └── Prediction2.ipynb        # Modélisation finale (gestion mémoire optimisée)
│
├── outputs/                     # Graphiques et résultats sauvegardés
├── requirements.txt             # Dépendances principales
└── README.md                    # Rapport complet
```

---

## 🔎 Partie 1 : Exploration des données

- Chargement de `title.basics.tsv` et `title.ratings.tsv`
- Visualisation initiale (`.head()`, `.info()`, `.describe()`)
- Nettoyage des valeurs manquantes (`NaN`) et suppression des doublons
- Conversion des colonnes numériques (`startYear`, `endYear`, `runtimeMinutes`)
- Filtrage des colonnes essentielles (`tconst`, `primaryTitle`, etc.)
- Réinitialisation des index et sauvegarde des jeux propres

Exemple :  

```python
title_basics = pd.read_csv("data/raw/title.basics.tsv", sep="\t", na_values="\N")
title_ratings = pd.read_csv("data/raw/title.ratings.tsv", sep="\t", na_values="\N")

dfs = {"title_basics": title_basics, "title_ratings": title_ratings}

for name, df in dfs.items():
    print(name, df.shape)
    print(df.head())
```

---

## 🤖 Partie 2 : Modélisation prédictive

### 1ère tentative (predict.ipynb)
- Objectif : Prédire `averageRating` avec **Régression Linéaire** et **Random Forest**  
- Problème : Impossible de charger **l’ensemble des données** → crash mémoire (RAM saturée).  

### 2ème version (Prediction2.ipynb)
- Chargement des données en **chunks** (200k lignes)
- Filtrage précoce (`startYear >= 2000`, `titleType == movie`, `numVotes >= 100`)
- Encodage :
  - **Genres** avec `MultiLabelBinarizer`
  - **titleType** avec `OneHotEncoder`
- Matrice finale : `(86682, 29)`  
- Modèles testés :
  - Régression Linéaire → `R² test ≈ 0.017`
  - Random Forest → `R² test ≈ 0.26` (meilleur mais surapprentissage)

Exemple d’évaluation :  

```python
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("R² test :", r2_score(y_test, y_pred))
```

---

## 📊 Résultats obtenus

- **Régression Linéaire** :  
  - `RMSE ≈ 1.35`, `R² test ≈ 0.017`  
  - Très faible pouvoir explicatif.

- **Random Forest** :  
  - `RMSE ≈ 1.17`, `R² test ≈ 0.26`  
  - Meilleures performances mais fort **écart entre train et test** → surapprentissage.

- **Features importantes (Random Forest)** :  
  - `numVotes`
  - `startYear`
  - certains genres (Drama, Comedy, Action…)

---

## ⚠️ Problèmes rencontrés & correctifs

### 🔹 Gestion des données (exploration)
- **Crash kernel** en chargeant tout `title.basics.tsv` → **Solution : lecture par chunks (200k lignes)**  
- **Erreurs de typage** (`Reality-TV` converti en numérique) → **Solution : `dtype=str` puis conversion ciblée en `numeric` avec `errors="coerce"`**  
- **Valeurs manquantes massives** → suppression ou filtrage (`dropna`, `fillna`)  
- **Doublons** → supprimés via `drop_duplicates()`

### 🔹 Problèmes de modélisation (prédiction)
- `scipy.sparse does not support dtype object` → conversion explicite en `float32` avant hstack  
- `LinearRegression` : erreur sur `squared=False` (ancienne version sklearn) → utilisation de `np.sqrt(mean_squared_error(...))`  
- Déséquilibre train/test → ajout de filtres (`numVotes >= 100`)  
- Surapprentissage RandomForest → test de `GridSearchCV` et `RandomizedSearchCV` pour limiter

### 🔹 Problèmes liés à pandas / pyarrow
- `ImportError: No module named 'pyarrow'` → installation via `pip install pyarrow`  
- `ValueError: The 'chunksize' option is not supported with the 'pyarrow' engine` → passage à `engine="c"`

### 🔹 Problèmes Git & Bash
- `nothing to commit, working tree clean` → aucun changement détecté  
- `error: failed to push some refs` → **Solution : `git pull --rebase origin main` avant push**  
- `fatal: pathspec '.gitignore' did not match any files` → fichier non créé, correction avec `touch .gitignore`  
- Rejet de push (erreur 403) → problème de droits, corrigé en configurant le bon remote et token GitHub

### 🔹 Difficultés diverses
- Crash RAM même après optimisation → nécessité de filtrer (`startYear >= 2000`, `titleType == movie`)  
- Visualisations lourdes → sauvegardées en PNG au lieu d’afficher en continu  
- Erreurs liées aux **versions de librairies** (`sklearn`, `scipy`, `pandas`) → mises à jour et ajustements

---

## 🚀 Améliorations futures
- Utiliser **LightGBM/XGBoost** pour améliorer la précision sans exploser la RAM  
- Intégrer un pipeline ML complet (`Pipeline` sklearn) avec `GridSearchCV` systématique  
- Stocker les données nettoyées dans un format **Parquet** pour chargement plus rapide  
- Tester une approche de **réduction de dimensionnalité** (PCA, SVD) avant les modèles  
- Automatiser via un script unique reproductible

---

## 🛠️ Installation & utilisation

### 1) Cloner le dépôt
```bash
git clone https://github.com/HDonzo/imdb-movie-analysis.git
cd imdb-movie-analysis
```

### 2) Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
```

### 3) Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4) Lancer Jupyter
```bash
jupyter notebook
```

---

## 📌 Auteurs

👤 **Donovan H.**  
Projet personnel d’exploration et de prédiction sur IMDB, réalisé avec Python (pandas, sklearn, matplotlib, seaborn, scipy).
