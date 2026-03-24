package Traitement

import org.apache.spark.sql.functions.*
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}

object Ingestion {

  def loadAndClean(spark: SparkSession, path: String): DataFrame = {

    val raw = spark.read
      .option("header", "true")
      .option("sep", ";")
      .option("encoding", "UTF-8")
      .csv(path)

    // Sélection uniquement des colonnes voulues
    val selected = raw.select(
      col("Identifiant station").as("station_id"),
      col("Nom de la Station").as("station_name"),
      col("Nom de la ligne").as("line_name"),
      col("Niveau de pollution").as("pollution_label"),
      col("stop_lon"),
      col("stop_lat"),
      col("niveau_pollution")
    ) 

    val cleaned =
      selected
        .withColumn("stop_lon",
          col("stop_lon").cast(DoubleType).alias("stop_lon"))
        // Convertir stop_lat en Double
        .withColumn("stop_lat",
          col("stop_lat").cast(DoubleType).alias("stop_lat"))
        // Retirer toutes les stations du RER
        .filter(!lower(col("line_name")).contains("rer"))
        // Retirer  toutes les lignes avec station aerienne
        .filter(!lower(col("pollution_label")).contains("aérienne"))
        // Retirer  toutes les lignes sans info sur la pollution
        .filter(!lower(col("pollution_label")).contains("données"))// Un seul enregistrement par station
        .dropDuplicates("station_id")
 
        // Remplacer seulement les champs vides
        .withColumn("station_name",
          when(trim(col("station_name")) === "" || col("station_name").isNull, lit("INCONNUE"))
            .otherwise(col("station_name")))
        .withColumn("line_name",
          when(trim(col("line_name")) === "" || col("line_name").isNull, lit("INCONNUE"))
            .otherwise(col("line_name")))
        .withColumn("pollution_label",
          when(trim(col("pollution_label")) === "" || col("pollution_label").isNull, lit("INCONNU"))
            .otherwise(col("pollution_label")))
    cleaned
  }
}
