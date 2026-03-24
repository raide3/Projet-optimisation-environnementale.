package Traitement

import org.apache.spark.sql.functions.*
import org.apache.spark.sql.{DataFrame, SparkSession}

object Transformations {

  /** Normalise le niveau de pollution en score numérique */
  def addPollutionScore(df: DataFrame): DataFrame = {
    df.withColumn(
      "pollution_score",
      when(lower(col("pollution_label")).contains("eleve"), 3)
        .when(lower(col("pollution_label")).contains("moyenne"), 2)
        .when(lower(col("pollution_label")).contains("faible"), 1)
        .otherwise(0)
    )
  }

  /** Exemple d'utilisation de map : convertir le nom en majuscules */
  def upperCaseStations(df: DataFrame): DataFrame =
    df.map(row =>
        row.getAs[String]("station_name").toUpperCase
      )(org.apache.spark.sql.Encoders.STRING)
      .toDF("station_name_uppercase")



}
