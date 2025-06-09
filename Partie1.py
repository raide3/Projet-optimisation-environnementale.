# ================================
# Analyse pollution - Réseau Métro IDF
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------
# 1. Nettoyage du dataset
# ------------------------------
df = pd.read_csv("qualite-de-lair-dans-le-reseau-de-transport-francilien.csv", sep=";")

# Supprimer lignes vides
df.dropna(how="all", inplace=True)

# Supprimer colonnes inutiles
colonnes_a_supprimer = [
    "Lien vers les mesures en direct", "Durée des mesures",
    "Mesures d’amélioration mises en place ou prévues",
    "point_geo", "pollution_air", "air", "niveau", "actions", "niveau_pollution"
]
df.drop(columns=[col for col in colonnes_a_supprimer if col in df.columns], inplace=True)

# Nettoyage des valeurs textuelles
df.replace("pas de données", pd.NA, inplace=True)

# Supprimer doublons
df.drop_duplicates(inplace=True)

# Supprimer lignes sans pollution mesurée
df.dropna(subset=["Niveau de pollution aux particules", "Niveau de pollution"], how="all", inplace=True)

# ------------------------------
# 2. ACP - Est-ce pertinent ?
# ------------------------------
"""
Non pertinent directement, car :
- La plupart des colonnes sont qualitatives ou textuelles
- Peu de variables numériques exploitables pour ACP
Une ACP nécessiterait d’encoder de nombreuses colonnes et de compléter beaucoup de valeurs manquantes.
"""

# ------------------------------
# 3. Filtrer stations métro et splitter
# ------------------------------
df_metro = df[df["Nom de la ligne"].str.contains("Métro", na=False)]

train, test = train_test_split(df_metro, test_size=0.3, random_state=42)
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

# ------------------------------
# 4. Modèle K-Means clustering
# ------------------------------
df_geo = df_metro[["stop_lat", "stop_lon"]].dropna()

kmeans = KMeans(n_clusters=3, random_state=42)
df_geo["cluster"] = kmeans.fit_predict(df_geo)

# Visualisation des clusters
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_geo, x="stop_lon", y="stop_lat", hue="cluster", palette="Set2")
plt.title("Clustering des stations de métro (K-Means)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("clusters_kmeans.png")
plt.close()

# ------------------------------
# 5. Modèle KNN
# ------------------------------
df_knn = df_metro[["stop_lat", "stop_lon", "Niveau de pollution"]].dropna()

# Encodage de la cible
le = LabelEncoder()
df_knn["pollution_encoded"] = le.fit_transform(df_knn["Niveau de pollution"])

X = df_knn[["stop_lat", "stop_lon"]]
y = df_knn["pollution_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# ------------------------------
# 6. Évaluation des modèles
# ------------------------------
print("=== Évaluation du modèle KNN ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# ------------------------------
# 7. Dashboard (à développer)
# ------------------------------
# Configuration
st.set_page_config(page_title="Pollution Métro IDF", layout="wide")

# Titre principal
st.title("🚇 Dashboard Pollution - Réseau Métro Île-de-France")

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv("metro_pollution_qgis.csv")
    df = df[df["Nom de la ligne"].str.contains("Métro", na=False)]
    df.replace("pas de données", pd.NA, inplace=True)
    return df.dropna(subset=["Niveau de pollution"])

df = load_data()

# Filtre latéral
lignes = df["Nom de la ligne"].dropna().unique()
ligne_selection = st.sidebar.multiselect("Sélectionner une ou plusieurs lignes de métro :", lignes, default=lignes)

df_filtered = df[df["Nom de la ligne"].isin(ligne_selection)]

# KPIs
st.subheader("📊 Indicateurs clés")
col1, col2, col3 = st.columns(3)
col1.metric("Nombre de stations", len(df_filtered))
col2.metric("Niveaux de pollution mesurés", df_filtered["Niveau de pollution"].nunique())
col3.metric("Lignes sélectionnées", len(ligne_selection))

# Carte interactive (si données géo)
st.subheader("🗺️ Carte des stations sélectionnées")
st.map(df_filtered.rename(columns={"stop_lat": "lat", "stop_lon": "lon"}))

# Graphiques
st.subheader("📉 Répartition des niveaux de pollution")
fig, ax = plt.subplots(figsize=(8,4))
sns.countplot(data=df_filtered, x="Niveau de pollution", order=df_filtered["Niveau de pollution"].value_counts().index, ax=ax)
st.pyplot(fig)

st.subheader("📈 Pollution par ligne de métro")
fig2, ax2 = plt.subplots(figsize=(10,5))
sns.countplot(data=df_filtered, y="Nom de la ligne", hue="Niveau de pollution", ax=ax2)
st.pyplot(fig2)

# Footer
st.markdown("---")
st.caption("Données : Île-de-France Mobilités • Visualisation : Streamlit • Projet : Qualité de l'air métro")

