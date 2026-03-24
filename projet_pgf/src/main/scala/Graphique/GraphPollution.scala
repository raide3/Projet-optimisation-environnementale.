package Graphique

import org.apache.spark.graphx.*
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.*

object GraphPollution {

  /** Convertit le DataFrame des stations en graphe GraphX */
  def buildGraph(df: DataFrame): Graph[(String, Double), String] = {

    val spark = df.sparkSession

    // 1. Créer les vertices
    val vertices = df.select("station_id", "station_name", "pollution_score")
      .rdd
      .map { row =>
        val id = row.getString(0).hashCode.toLong
        (id, (row.getString(1), row.getDouble(2))) 
      }

    // 2. Créer les arêtes (chaîne de stations par ligne)
    val edges = df
      .withColumn("fake_order", monotonically_increasing_id())
      .orderBy("line_name", "fake_order")
      .select("station_id", "line_name")
      .rdd
      .map(row => (row.getString(1), row.getString(0)))
      .groupByKey()
      .flatMap { case (_, stationIds) =>
        val list = stationIds.toList
        list.sliding(2).map { case Seq(a, b) =>
          Edge(a.hashCode.toLong, b.hashCode.toLong, "CONNECTED")
        }
      }

    Graph(vertices, edges)
  }

  /** Calcul de la pollution propagée dans le réseau. Elle prend les scores de pollution existants 
   * dans les stations (srcAttr._2) et envoie un pourcentage (0.4 ici) à ses voisins */
  def computePropagation(graph: Graph[(String, Double), String]): VertexRDD[Double] = {
    graph.aggregateMessages[Double](
      triplet => {
        // le noeud pollué augmente la pollution du voisin
        val propagation = triplet.srcAttr._2 * 0.4
        triplet.sendToDst(propagation)
      },
      _ + _
    )
  }
}

