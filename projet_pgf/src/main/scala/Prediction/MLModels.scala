package Prediction

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.DataFrame

object MLModels {

  def trainClassification(dfFeatures: DataFrame): org.apache.spark.ml.regression.RandomForestRegressionModel = {
    // Divise les données : 80% entraînement, 20% test (seed fixe pour reproductibilité)
    val Array(train, test) = dfFeatures.randomSplit(Array(0.8, 0.2), seed = 42)

    // Configure le régresseur RandomForest
    val rf = new RandomForestRegressor()
      .setLabelCol("pollution_score")
      .setFeaturesCol("features")
      .setNumTrees(50)
      .setMaxBins(1000)

    // Entraîne le modèle sur les données d'entraînement
    val model = rf.fit(train)
    // Teste le modèle sur les données de test
    val predictions = model.transform(test)
    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    println(s"Classification accuracy = ${eval.evaluate(predictions)}")
    model
  }
  
}
