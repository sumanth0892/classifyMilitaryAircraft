#Create a pipeline and classify the images of airplanes
import os
from functools import reduce
from sparkdl import readImages
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml.image import ImageSchema

#Build a spark session
def getData():
	spark = SparkSession.builder.appName('Military Aircraft Classifier').getOrCreate()
	fPath = "....../dataSets/militaryAircraft/mainData"
	imageDFrames = []; i = 0
	for target in sorted(os.listdir(fPath)):
		df = readImages(os.path.join(fPath,target)).withColumn("label",lit(i))
		imageDFrames.append(df)
		i += 1
	#Now, we have data frames with images. 
	#Split this into test and train datasets and merge them to run a classification model
	trainSplits = []; testSplits = []
	for df in imageDFrames:
		train, test = df.randomSplit([0.75, 0.25])
		trainSplits.append(train)
		testSplits.append(test)
	trainDF = train[0]; testDF = test[0]
	trainSplits.pop(0); testSplits.pop(0)
	for dfTrain,dfTest in zip(trainSplits,testSplits):
		trainDF.unionAll(dfTrain)
		testDF.unionAll(dfTest)
	#Free up some memory
	del train,test,trainSplits,testSplits
	return trainDF,testDF



