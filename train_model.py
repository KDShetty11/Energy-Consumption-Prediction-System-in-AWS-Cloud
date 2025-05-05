import numpy as np
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.evaluation import RegressionMetrics
import warnings

warnings.filterwarnings("ignore")

# Spark setup
conf = SparkConf().setAppName('EnergyConsumptionPrediction').setMaster('local')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("ERROR")


print("Loading dataset...")
df = spark.read.csv("TrainingDataset.csv", header=True, inferSchema=True)
for column in df.columns:
    df = df.withColumnRenamed(column, column.replace(' ', '_'))

# Identify categorical columns
categorical_columns = [col_name for col_name, dtype in df.dtypes if dtype == 'string']
print(f"Categorical columns identified: {categorical_columns}")

# Index categorical columns
for column in categorical_columns:
    unique_values = df.select(column).distinct().rdd.flatMap(lambda x: x).collect()
    mapping_dict = {val: idx for idx, val in enumerate(sorted(unique_values))}
    map_udf = udf(lambda val: mapping_dict.get(val, -1), IntegerType())
    df = df.withColumn(f"{column}_indexed", map_udf(col(column)))

# Drop original categorical columns
df_final = df.drop(*categorical_columns)

# Cast all columns to float
for col_name in df_final.columns:
    df_final = df_final.withColumn(col_name, col(col_name).cast('float'))

energy_column = 'Energy_Consumption'
feature_columns = [col_name for col_name in df_final.columns if col_name != energy_column]

# Prepare data for MLlib
def row_to_labeled_point(row):
    features = [row[col] for col in feature_columns]
    return LabeledPoint(row[energy_column], features)

data_rdd = df_final.rdd.map(row_to_labeled_point)

# Split data
train_rdd, test_rdd = data_rdd.randomSplit([0.7, 0.3], seed=21)

print("Training Gradient Boosted Trees regressor...")
model = GradientBoostedTrees.trainRegressor(
    train_rdd,
    categoricalFeaturesInfo={}, 
    numIterations=125,
    learningRate=0.2,
    maxDepth=2
)

print("Evaluating model...")
predictions = model.predict(test_rdd.map(lambda x: x.features))
predictionAndLabel = predictions.zip(test_rdd.map(lambda x: x.label))
metrics = RegressionMetrics(predictionAndLabel)

# Visually pleasing output
print("\n" + "="*40)
print("   Energy Consumption Prediction Results")
print("="*40)
print(f"Root Mean Squared Error (RMSE): {metrics.rootMeanSquaredError:.2f}")
print(f"R2 (coefficient of determination): {metrics.r2:.4f}")
print("="*40 + "\n")

model.save(sc, 'regression_model')
print("Model saved successfully.")