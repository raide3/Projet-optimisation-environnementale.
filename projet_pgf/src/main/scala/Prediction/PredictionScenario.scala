package Prediction
import Graphique.*
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions.*
import org.apache.spark.sql.{DataFrame, SparkSession}

object PredictionScenario {

  def run(df: DataFrame, spark: SparkSession): Unit = {

    println("\n Ajout de la station fictive NEW_0010")
    val added = NewStation.addVirtualStation(df, spark)

    println(" Rechargement du modèle ML...")
    val model = PipelineModel.load("model/pollution_model")

    println(" Application du pipeline ML...")
    val dfML = model.transform(added)

    // Prédiction ML pour NEW_001
    val predML = dfML.filter(col("station_id") === "NEW_0010")
      .select("prediction")
      .head()
      .getDouble(0)

    println(s"➡ Prédiction pollution pour NEW_008 = $predML (1=FAIBLE, 2=MOYENNE, 3=FORTE)")

    println(" Calcul de la propagation via GraphX...")

    val graph = GraphPollution.buildGraph(dfML)
    val propagated = GraphPollution.computePropagation(graph)

    val newId = "NEW_0010".hashCode.toLong
    val prop = propagated.lookup(newId).headOption.getOrElse(0.0)

    println(s"➡ Pollution propagée : $prop")

    // Score final combiné
    val finalScore = predML

    println("\n Résultat FINAL pour NEW_0010")
    println(s" Score pollution global = $finalScore\n")
  }
}
