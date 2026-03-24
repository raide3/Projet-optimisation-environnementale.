import Graphique.GraphPollution
import Prediction.{PredictionScenario, TrainAndSaveModel}
import Traitement.{Analysis, Ingestion, Transformations}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.*
import Analysis2.*

object main {

  def main(args: Array[String]): Unit = {

    // 1. Création de la session Spark
    val spark = SparkSession.builder()
      .appName("Projet Pollution Spark")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
      .config("spark.kryo.registrationRequired", "false")
      .config("spark.kryo.referenceTracking", "false")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    println("\n --------------------------------DÉMARRAGE DU PROJET SPARK-----------------------------------------------\n")

    // 2. Chargement et nettoyage du dataset
    val df = Ingestion.loadAndClean(
      spark,
      path = "data/Pollution.csv"
    )

    println("\n --------------------------------Données après nettoyage : ------------------------------------------------")
    df.show(10, truncate = false)

    // 3. Ajout du score de pollution dans la colonne PollutionScore
    val scored = Transformations.addPollutionScore(df)

    // 4.  pour la cohérence il faut utiliser des valeurs réelles ML : stop_lon / stop_lat → double
    val scoredDoubles = scored
      .withColumn("stop_lon", col("stop_lon").cast("double"))
      .withColumn("stop_lat", col("stop_lat").cast("double"))
      .withColumn("pollution_score", col("pollution_score").cast("double"))

    // Les differents stats par lignes et puis les 10 lignes les plus polluées

    println("\n--------------------------------------- Statistiques de pollution par ligne : ----------------------------------")
    Analysis
      .pollutionStatsByLine(scoredDoubles)
      .show(truncate = false)

    println("\n--------------------------------------- TOP 10 stations les plus polluées :-----------------------------------------------")
    Analysis
      .topStations(scoredDoubles)
      .show(truncate = false)
    // analyse de la pollution pour une station particulière (Châtelet)
    val newDatasetPath = "data/qualite-de-lair-chatelet.csv"
    StatAnalysis.run(spark, newDatasetPath)
    // les annomalies détectés ie les colonnes n'ayant pas de valeurs dans la col PollutionScore qui sera utilisé lors de la prédiction
    println("\n------------------------------------------⚠ ANOMALIES détectées :----------------------------------------------------------")
    Analysis
      .detectAnomalies(scoredDoubles)
      .show(truncate = false)

    println("\n---------------------------------------------✔ PIPELINE TERMINÉ------------------------------------------------------------------------\n")


    println("\n-------------------------------------------------- Construction du graphe pollution... -----------------------------------------------")

    // construction du graphe grêce à graphx puis affichage du nombre de noeuds d'aretes et la propagation de pollution issu des stations voisinnes (10)
    val graph = GraphPollution.buildGraph(scoredDoubles)

    graph.triplets.take(10).foreach { triplet => println(s"${triplet.srcAttr} ---> ${triplet.dstAttr}") }

    println(s"\n➡ Nombre de stations = ${graph.numVertices}")
    println(s"➡ Nombre de connexions = ${graph.numEdges}")

    println("\n Propagation calculée :")
    //Sous la forme :le sommet (station hash 123456789) reçoit 1.6 points de pollution ➡️ à cause des stations voisines polluées
    GraphPollution
      .computePropagation(graph)
      .collect()
      .take(10)
      .foreach(println)



    // ------------ MACHINE LEARNING ---------------------
    //Vérifie si un modele de prediction a déjà été calculé sinon démarre le test et l'apprentissage
    val modelPath = "model/pollution_model"

    val model =
      if (new java.io.File(modelPath).exists()) {
        println("\n Modèle existant trouvé → chargement...")
        PipelineModel.load(modelPath)
      } else {
        println("\n Aucun modèle trouvé → entraînement...")
        TrainAndSaveModel.run(scoredDoubles, modelPath)
      }

    //Déroule le scénario d'ajout d'une station et calcule sa pollution en fontion de sa ligne et sa position et calcule aussi la part issue da la pollution progagée
    println("\n-----------------------------------Scénario : prédiction pour une nouvelle station...-----------------------------------")
    PredictionScenario.run(scoredDoubles, spark)

    spark.stop()
  }

}
