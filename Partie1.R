# ==============================
# Analyse pollution métro RATP
# ==============================

# Chargement des packages
library(dplyr)
library(readr)
library(ggplot2)
library(caret)
library(class)
library(cluster)
set.seed(123)

# 1. Nettoyage du dataset
df <- read_csv2("qualite-de-lair-dans-le-reseau-de-transport-francilien.csv")

# Supprimer lignes vides
df <- df[rowSums(is.na(df)) != ncol(df), ]

# Colonnes inutiles à supprimer
colonnes_a_supprimer <- c(
  "Lien vers les mesures en direct", 
  "Durée des mesures", 
  "Mesures d’amélioration mises en place ou prévues", 
  "point_geo", "pollution_air", "air", "niveau", "actions", "niveau_pollution"
)
df <- df %>% select(-one_of(intersect(colnames(df), colonnes_a_supprimer)))

# Remplacer "pas de données" par NA
df[df == "pas de données"] <- NA

# Supprimer doublons
df <- distinct(df)

# Supprimer lignes sans pollution mesurée
df <- df %>% filter(!(is.na(`Niveau de pollution aux particules`) & is.na(`Niveau de pollution`)))

# ==========================================
# 2. ACP : Est-elle intéressante ?
# ==========================================
# Non, car la majorité des variables sont qualitatives.
# Pour faire une ACP pertinente, il faut :
# - Des variables numériques continues
# - Peu de valeurs manquantes
# Donc une ACP n’est pas directement applicable sauf après encodage très complet et nettoyage avancé.

# ==========================================
# 3. Filtrer métro et créer train/test
# ==========================================
df_metro <- df %>% filter(grepl("Métro", `Nom de la ligne`))

# Sauvegarde train/test
train_idx <- sample(1:nrow(df_metro), 0.7 * nrow(df_metro))
train <- df_metro[train_idx, ]
test <- df_metro[-train_idx, ]

write_csv(train, "train.csv")
write_csv(test, "test.csv")

# ==========================================
# 4. K-means pour regrouper les stations
# ==========================================
# Encodage simplifié : on ne garde que les colonnes géographiques
df_metro_geo <- df_metro %>% select(stop_lat, stop_lon) %>% drop_na()

# K-means clustering (3 groupes arbitrairement)
kmeans_model <- kmeans(df_metro_geo, centers = 3, nstart = 10)

# Ajouter les clusters au dataset
df_metro_geo$cluster <- as.factor(kmeans_model$cluster)
ggplot(df_metro_geo, aes(x = stop_lon, y = stop_lat, color = cluster)) +
  geom_point() +
  labs(title = "Clustering des stations métro par K-means")

# ==========================================
# 5. KNN : prédire niveau de pollution
# ==========================================
# Préparation : transformer variable à prédire en facteur
df_knn <- df_metro %>% 
  filter(!is.na(`Niveau de pollution`)) %>%
  select(stop_lat, stop_lon, `Niveau de pollution`)

df_knn$`Niveau de pollution` <- as.factor(df_knn$`Niveau de pollution`)

# Split
idx <- sample(1:nrow(df_knn), 0.7 * nrow(df_knn))
train_knn <- df_knn[idx, ]
test_knn <- df_knn[-idx, ]

# KNN
pred_knn <- knn(train = train_knn[, c("stop_lat", "stop_lon")],
                test = test_knn[, c("stop_lat", "stop_lon")],
                cl = train_knn$`Niveau de pollution`,
                k = 3)

# ==========================================
# 6. Évaluation des modèles
# ==========================================
# Évaluation KNN
confusionMatrix(pred_knn, test_knn$`Niveau de pollution`)

# Évaluation K-means : pas une prédiction, mais un regroupement
# => Peut être visualisé et comparé à des zones géographiques
