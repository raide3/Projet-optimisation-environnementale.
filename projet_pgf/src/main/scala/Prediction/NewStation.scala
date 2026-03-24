package Prediction

import org.apache.spark.sql.functions.*
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object NewStation {

  def addVirtualStation(df: DataFrame, spark: SparkSession): DataFrame = {
    // On récupère le schema exact du df
    val schema: StructType = df.schema

    // Création d'une ligne compatible
    val rows = Seq(
      Row(
        "NEW_0010",                     
        "Station Test10",         
        "Métro 13",                  
        "INCONNU",
        2.3201902468999456,
        48.89062201554889,                      
        "non mesuré",
        0.0// niveau_pollution
      )
    )

    val rdd = spark.sparkContext.parallelize(rows)
    val newDf = spark.createDataFrame(rdd, schema)

    // Union
    df.unionByName(newDf)
      .withColumn("stop_lon", col("stop_lon").cast("double"))
      .withColumn("stop_lat", col("stop_lat").cast("double"))
  }
}
