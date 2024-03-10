from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

# Spark Session 
spark = SparkSession.builder.appName("FlightPricePrediction").getOrCreate()
df_pyspark = spark.read.csv('C:\\Users\\beyza\\OneDrive\\Masaüstü\\Clean_Dataset.csv', header=True, inferSchema=True)

categorical_features = ['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(df_pyspark) for col in categorical_features]
pipeline = Pipeline(stages=indexers)
df_pyspark_indexed = pipeline.fit(df_pyspark).transform(df_pyspark)

# Gereksiz sutunlari cikar
selected_features = ['duration', 'days_left', 'price'] + [col+"_index" for col in categorical_features]
df_pyspark_final = df_pyspark_indexed.select(selected_features)

# ozellikleri bi araya getir
assembler = VectorAssembler(inputCols=['duration', 'days_left'] + [col+"_index" for col in categorical_features], outputCol='features')
df_pyspark_assembled = assembler.transform(df_pyspark_final)

# egitim ve test veri setleri bol
train_data, test_data = df_pyspark_assembled.randomSplit([0.8, 0.2], seed=123)

# Lineer regresyon modeli ve egit
lr = LinearRegression(featuresCol='features', labelCol='price')
model = lr.fit(train_data)

# predict
predictions = model.transform(test_data)
predictions.select("prediction", "price", "features").show()

# Model performansi
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")


import matplotlib.pyplot as plt
import pandas as pd

# Prediction, price ve features sütunlarını seç
predictions_pd = predictions.select("prediction", "price", "features").toPandas()

# Gerçek fiyatlar ile tahminleri görselleştir
plt.scatter(predictions_pd["price"], predictions_pd["prediction"])
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Edilen Fiyatlar")
plt.title("Gerçek vs. Tahmin Edilen Fiyatlar")
plt.show()



spark.stop()
