%python 

from pyspark.sql.functions import udf

from pyspark.sql.types import IntegerType

from pyspark.sql import SparkSession

from pyspark.ml.regression import LinearRegression

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml import Pipeline

import numpy as np

#Create a Spark session
spark = SparkSession.builder.appName("MLlibExample").getOrCreate()

#Generate a toy dataset for illustration
np.random.seed(42) 
num_samples = 1000

#Features: number of bedrooms, square footage
data = [(np.random.randint(1, 5), 100 + 50 * np.random.rand(), 150 + 75 * np.random.randint(1, 5) + 0.1 * (100 + 50 * np.random.rand()) + 10 * np.random.randn()) for _ in range(num_samples)]

#Create a DataFrame
df = spark.createDataFrame(data, ["bedrooms", "square_footage", "price"])

#Create a feature vector
feature_cols = ["bedrooms", "square_footage"] 
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="my_features")

#Rename output column "features" to a new name "my_features"
df = vector_assembler.transform(df).withColumnRenamed("my_features", "features")

#Split the data into training and testing sets
(train_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

#Build a Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="price")

#Create a pipeline
pipeline = Pipeline(stages=[vector_assembler, lr])

#Train the model
model = pipeline.fit(train_data)

#Make predictions on the test set
predictions = model.transform(test_data)

#Evaluate the model
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="mse") 
mse = evaluator.evaluate(predictions)

print(f"Mean Squared Error on Test Set: {mse}")