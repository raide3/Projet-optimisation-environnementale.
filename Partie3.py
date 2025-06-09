import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph

# -----------------------
# 1. Chargement des données
# -----------------------
# Fichier contenant station1, station2, pollution
df_edges = pd.read_csv("station_connections.csv")
df_pollution = {
    "A": 3.5, "B": 2.0, "C": 5.0, "D": 3.0, "E": 4.5, "F": 2.5  # pollution fictive
}

# -----------------------
# 2. Construction du graphe pondéré
# -----------------------
G = nx.Graph()
for _, row in df_edges.iterrows():
    u, v = row["station1"], row["station2"]
    weight = (df_pollution.get(u, 3.0) + df_pollution.get(v, 3.0)) / 2
    G.add_edge(u, v, weight=weight)

# Affectation du signal pollution
for node in G.nodes():
    G.nodes[node]["pollution"] = df_pollution.get(node, np.nan)

# -----------------------
# 3. Construction de la matrice Laplacienne
# -----------------------
L = nx.laplacian_matrix(G, weight="weight")
A = nx.adjacency_matrix(G, weight="weight")

# Optimisation : matrice creuse utilisée via scipy.sparse

# -----------------------
# 4. Analyse spectrale
# -----------------------
# Valeurs propres/vecteurs propres : on cherche les plus petits (diffusion lente)
eigvals, eigvecs = eigsh(L.asfptype(), k=4, which='SM')  # k plus petites valeurs propres

# Visualisation du 2e vecteur propre (le premier est constant = mode trivial)
fiedler_vector = eigvecs[:, 1]

# -----------------------
# 5. Visualisation de la propagation
# -----------------------
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_size=600,
        node_color=fiedler_vector, cmap=plt.cm.plasma, edge_color='gray')
plt.title("🔍 Analyse spectrale de la pollution – Fiedler vector")
plt.colorbar(label="Valeur spectrale (intensité relative)")
plt.savefig("spectral_pollution_propagation.png")
plt.close()

# -----------------------
# 6. Interprétation
# -----------------------
max_pollution_zone = np.argmax(fiedler_vector)
most_exposed_station = list(G.nodes())[max_pollution_zone]

print("📍 Station probablement la plus exposée à la pollution :", most_exposed_station)
