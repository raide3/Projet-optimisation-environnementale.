import pandas as pd
import networkx as nx
import os

# =============================================
# 1. Construction du graphe à partir d'un CSV
# =============================================

# Hypothèse : vous avez un dataset avec des trajets explicites ou implicites.
# Comme ce n’est pas fourni, on simule un jeu de données ici :
data = {
    "station1": ["A", "A", "B", "C", "C", "D", "E"],
    "station2": ["B", "C", "D", "D", "E", "F", "A"],  # exemple de boucle possible : A -> C -> E -> A
}

df_links = pd.DataFrame(data)
df_links["connected"] = True  # Booléen toujours vrai ici

# Suppression doublons
df_links = df_links.drop_duplicates()

# Sauvegarde CSV
df_links.to_csv("station_connections.csv", index=False)

# Taille du fichier
file_size = os.path.getsize("station_connections.csv") / 1024  # en Ko
print(f"📦 Taille du fichier CSV : {file_size:.2f} Ko")

# ➤ Réduction possible : remplacer les noms par des IDs courts (A → 0, B → 1...) ou compresser le fichier (gzip)

# Construction du graphe
G = nx.from_pandas_edgelist(df_links, "station1", "station2")

# Afficher le graphe (optionnel)
# nx.draw(G, with_labels=True)

# =============================================
# 2. Chemin minimisant l’exposition à la pollution
# =============================================

# Hypothèse : pollution stockée par station, + temps estimé entre stations
pollution_data = {
    "A": 3, "B": 2, "C": 5, "D": 2, "E": 4, "F": 1
}
time_data = {
    ("A", "B"): 3, ("A", "C"): 5, ("B", "D"): 2, ("C", "D"): 3,
    ("C", "E"): 4, ("D", "F"): 2, ("E", "A"): 3
}

# Ajouter pondération pollution et temps
for (u, v) in G.edges():
    G[u][v]["pollution"] = (pollution_data[u] + pollution_data[v]) / 2
    G[u][v]["time"] = time_data.get((u, v), time_data.get((v, u), 10))  # valeur par défaut

# Multi-critère : ici on minimise la somme pollution + lambda*temps
def path_min_pollution_time(graph, source, target, lambda_time=1.0):
    for u, v, d in graph.edges(data=True):
        d["cost"] = d["pollution"] + lambda_time * d["time"]
    return nx.shortest_path(graph, source=source, target=target, weight="cost")

print("📍 Chemin optimisé (pollution + temps) de A à F :", path_min_pollution_time(G, "A", "F", lambda_time=0.8))

# =============================================
# 3. Chemin avec seuil de pollution
# =============================================

def path_with_pollution_threshold(graph, source, target, threshold):
    # Filtrer les sommets par seuil
    valid_nodes = [n for n in graph.nodes if pollution_data.get(n, float("inf")) <= threshold]
    subgraph = graph.subgraph(valid_nodes)
    try:
        path = nx.shortest_path(subgraph, source=source, target=target)
        return path
    except nx.NetworkXNoPath:
        return None

print("🔎 Chemin sous seuil pollution 3 entre A et F :", path_with_pollution_threshold(G, "A", "F", threshold=3))

# =============================================
# 4. Détection de cycles
# =============================================

def detect_cycles(graph):
    try:
        cycles = list(nx.simple_cycles(nx.DiGraph(graph)))
        return cycles
    except:
        return []

cycles = detect_cycles(G)
print("🔁 Cycles détectés :", cycles)

"""
Justification :
- `networkx.simple_cycles()` (algorithme de Johnson) détecte tous les cycles simples.
- Complexité : O((n + e)(c + 1)), où c = nombre de cycles.
Intérêt :
- Détection de boucles qui augmentent le temps de trajet ou la pollution
- Utile pour optimiser la ventilation ou la signalétique
- Utile pour diagnostiquer des erreurs ou redondances dans les données
"""
