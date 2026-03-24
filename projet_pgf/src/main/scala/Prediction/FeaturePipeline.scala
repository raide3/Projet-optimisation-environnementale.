package Prediction

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.*
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

object FeaturePipeline {


    def buildPipelineOnly(df: DataFrame): Pipeline = {
      // Convertit les labels textuels (ex: "FORTE") en indices numériques (1,2...)
      val labelIndexer = new StringIndexer()
        .setInputCol("pollution_label")
        .setOutputCol("label")
        .setHandleInvalid("keep")

      // Convertit les noms de stations en indices numériques (catégorie -> nombre)
      val stationIndexer = new StringIndexer()
        .setInputCol("station_name")
        .setOutputCol("station_idx")
        .setHandleInvalid("keep")

      // Convertit les noms de lignes (Métro 5, RER A...) en indices numériques  
      val lineIndexer = new StringIndexer()
        .setInputCol("line_name")
        .setOutputCol("line_idx")
        .setHandleInvalid("keep")

      // Transforme les indices de ligne en vecteur one-hot (ex: [0,1,0,0,1])
      val lineEncoder = new OneHotEncoder()
        .setInputCol("line_idx")
        .setOutputCol("line_vec")

      // Assemble toutes les features en un seul vecteur pour le modèle ML
      val assembler = new VectorAssembler()
        .setInputCols(Array("station_idx", "line_vec", "stop_lon", "stop_lat"))
        .setOutputCol("features")

      // Crée le pipeline avec toutes les étapes dans l'ordre 

      new Pipeline().setStages(Array(labelIndexer, stationIndexer, lineIndexer, lineEncoder, assembler))
    }


}
