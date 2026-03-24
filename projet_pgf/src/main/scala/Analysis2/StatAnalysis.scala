package Analysis2

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object StatAnalysis {

  def run(spark: SparkSession, filePath: String): Unit = {
    println("\n--- Démarrage de l'Analyse Statistique et du Nettoyage ---")

    // 1. Chargement du Dataset
    var df = spark.read
      .option("header", "true")
      .option("delimiter", ";")
      .option("locale", "fr-FR")
      .option("inferSchema", "true")
      .csv(filePath)
      .withColumnRenamed("date/heure", "date_heure")

    println(s"Total d'enregistrements chargés : ${df.count()}")

    // --- ÉTAPE DE NETTOYAGE & CASTING ---

    df = df.withColumnRenamed("PM10", "PM10_str")
      .withColumnRenamed("TEMP", "TEMP_str")
      .withColumnRenamed("HUMI", "HUMI_str")

    // 2. Conversion Manuelle des Types Numériques (Gestion de 'ND' et de la virgule décimale)
    df = df.withColumn("PM10_clean",
        regexp_replace(regexp_replace(col("PM10_str"), "ND", ""), ",", ".").try_cast(DoubleType)) // 💡 TRY_CAST
      .withColumn("TEMP_clean",
        regexp_replace(regexp_replace(col("TEMP_str"), "ND", ""), ",", ".").try_cast(DoubleType)) // 💡 TRY_CAST
      .withColumn("HUMI_clean",
        regexp_replace(regexp_replace(col("HUMI_str"), "ND", ""), ",", ".").try_cast(DoubleType)) // 💡 TRY_CAST

    // Remplacer les anciennes colonnes par les nouvelles nettoyées
    df = df.drop("PM10", "TEMP", "HUMI", "PM10_str", "TEMP_str", "HUMI_str")
      .withColumnRenamed("PM10_clean", "PM10")
      .withColumnRenamed("TEMP_clean", "TEMP")
      .withColumnRenamed("HUMI_clean", "HUMI")

    // 3. Conversion de la Date/Heure
    df = df.withColumn("timestamp",
        to_timestamp(col("date_heure"), "yyyy-MM-dd'T'HH:mm:ssXXX")
      )
      .drop("date_heure")

    // 4. Gestion des Valeurs Manquantes et Aberrantes
    val initialCount = df.count()
    // Supprime les lignes où les valeurs PM10, TEMP, HUMI ou timestamp sont nulles
    df = df.na.drop(Seq("PM10", "TEMP", "HUMI", "timestamp"))
    // Filtre les valeurs extrêmes de PM10 (erreurs de capteur)
    df = df.filter(col("PM10") >= 0 and col("PM10") < 500)

    val cleanedCount = df.count()
    println(s"Valeurs Manquantes et Aberrantes supprimées : ${initialCount - cleanedCount} lignes")
    println(s"Total d'enregistrements après nettoyage : ${df.count()}")

    // --- ÉTAPE D'ANALYSE STATISTIQUE ---

    println("\n---  Statistiques Descriptives (PM10, TEMP, HUMI) ---")
    df.select("PM10", "TEMP", "HUMI").describe().show()

    println("\n---  Tendance Temporelle (Moyenne Mensuelle de PM10) ---")
    df.withColumn("Mois", date_format(col("timestamp"), "yyyy-MM"))
      .groupBy("Mois")
      .agg(avg("PM10").alias("PM10_Moyen_Mensuel"))
      .sort("Mois")
      .show()

    println("\n--- Corrélation : PM10 vs TEMP & PM10 vs HUMI ---")

    val corrTemp = df.stat.corr("PM10", "TEMP")
    val corrHumi = df.stat.corr("PM10", "HUMI")

    println(f"Corrélation (PM10 vs TEMP) : $corrTemp%.4f")
    println(f"Corrélation (PM10 vs HUMI) : $corrHumi%.4f")

    println("\n--- Analyse Statistique Terminée ---")
  }
}