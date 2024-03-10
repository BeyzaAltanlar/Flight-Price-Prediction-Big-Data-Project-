from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

def detect_problematic_categorical_features( df_pyspark,threshold=32):
    problematic_features = []

    for column in df_pyspark.schema:
        if "StringType" in str(column.dataType):
            distinct_count = df_pyspark.select(col(column.name)).distinct().count()
            if distinct_count > threshold:
                problematic_features.append(column.name)

    return problematic_features

problematic_features = detect_problematic_categorical_features(df_pyspark)
categorical_features_dropped = [col for col in categorical_features if col not in problematic_features]

df_pyspark_dropped = df_pyspark.select(['duration', 'days_left', 'price'] + categorical_features_dropped)


indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(df_pyspark_dropped) for col in categorical_features_dropped]
pipeline = Pipeline(stages=indexers)
df_pyspark_indexed = pipeline.fit(df_pyspark_dropped).transform(df_pyspark_dropped)

# Gereksiz sutunlari cikar
selected_features = ['duration', 'days_left', 'price'] + [col+"_index" for col in categorical_features_dropped]
df_pyspark_final = df_pyspark_indexed.select(selected_features)

# ozellikleri bi araya getir
assembler = VectorAssembler(inputCols=['duration', 'days_left'] + [col+"_index" for col in categorical_features_dropped], outputCol='features')
df_pyspark_assembled = assembler.transform(df_pyspark_final)

# egitim ve test veri setleri bol
(train_data, test_data) = df_pyspark_assembled.randomSplit([0.7, 0.3])

# Lineer regresyon modeli ve egit
classifier = DecisionTreeRegressor(labelCol = 'price', featuresCol = 'features')
#pipeline = Pipeline(stages= indexers + [assembler, classifier])
model = classifier.fit(train_data)

# predict
predictions = model.transform(test_data)
predictions.select("prediction", "price", "features").show()

# Model performansi
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")