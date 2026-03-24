package Prediction

import Prediction.{FeaturePipeline, MLModels}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame

object TrainAndSaveModel {

  def run(dfFeatures: DataFrame, modelPath: String): PipelineModel = {

    println(s"\n------------------------- Entraînement du modèle ML et sauvegarde dans : $modelPath----------------------------------")

    // 1. Construction pipeline NON fit
    val pipeline = FeaturePipeline.buildPipelineOnly(dfFeatures) // <-- nouvelle méthode qui retourne juste le Pipeline

    // 2. Fit pipeline sur dfFeatures pour transformer les colonnes
    val transformed = pipeline.fit(dfFeatures).transform(dfFeatures)

    // 3. Entraînement RandomForest sur les features assemblées
    val rfModel = MLModels.trainClassification(transformed)

    // 4. Pipeline complet = pipeline + RF
    val finalPipeline = new Pipeline()
      .setStages(pipeline.getStages :+ rfModel)
    val pipelineModel = finalPipeline.fit(dfFeatures)

    // 5. Sauvegarde
    pipelineModel.write.overwrite().save(modelPath)
    println(s"✔ Modèle sauvegardé sous : $modelPath")
    pipelineModel
  }


}
