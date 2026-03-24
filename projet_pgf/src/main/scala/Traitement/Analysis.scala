package Traitement

import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions.*
import org.apache.spark.sql.{DataFrame, SparkSession}


object Analysis {

  // Top 10 stations les plus polluées
  def topStations(df: DataFrame): DataFrame =
    df.orderBy(col("pollution_score").desc).limit(10)

  /** Détection d'anomalies :
   * pollution incohérente (score 0 = non défini alors que pollution_air existe)
   */
  def detectAnomalies(df: DataFrame): DataFrame =
    df.filter(
      col("pollution_score") === 0 &&
        col("pollution_air").isNotNull
    )

  /** Création d'un score global géospatial
   * pollution_score / coeff arbitraire + distance point le plus pollué
   */

  val refLon = 2.3522
  val refLat = 48.8566

  val distanceUDF = udf((lon: Double, lat: Double) => {
    math.sqrt(math.pow(lon - refLon, 2) + math.pow(lat - refLat, 2))
  }, DoubleType)
  
  def pollutionGeoIndex(df: DataFrame): DataFrame =
    df.withColumn("geo_index", col("pollution_score") / (lit(1.0) + distanceUDF(col("stop_lon"), col("stop_lat"))))

  /** Stats pollution par ligne */
  def pollutionStatsByLine(df: DataFrame): DataFrame =
    df.groupBy("line_name")
      .agg(
        avg("pollution_score").as("avg_pollution"),
        max("pollution_score").as("max_pollution"),
        min("pollution_score").as("min_pollution"),
      )
}
