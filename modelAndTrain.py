#Create a model and train the data
import os
import transformData as tD
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.classification import LogisticRegression

if __name__ == '__main__':
	featurizer = DeepImageFeaturizer(inputCol = "image",outputCol = "features",modelName = "InceptionV3")
	lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
	p = Pipeline(stages = [featurizer,lr])
	#Import the data
	train,test = main.getData()
	p_model = p.fit(train)
	predictions = p_model.transform(test)
	predictions.select("filePath", "prediction").show(truncate=False)

	#Evaluation
	from pyspark.ml.evaluation import MultiClassificationEvaluator
	df = p_model.transform(test)
	df.show()
	predictionAndLabels = df.select("prediction", "label")
	evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
	print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))



