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

# -------------------- Spark Setup --------------------
print("\n[STATUS] Initializing Spark Context and Session...")
conf = SparkConf().setAppName('EnergyConsumptionPrediction').setMaster('local')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("ERROR")

# -------------------- Data Loading --------------------
print("[STATUS] Loading dataset...")
df = spark.read.csv("TrainingDataset.csv", header=True, inferSchema=True)

# Clean column names (replace spaces with underscores)
for column in df.columns:
    df = df.withColumnRenamed(column, column.replace(' ', '_'))

print("[STATUS] Dataset schema:")
df.printSchema()
print(f"[STATUS] Total rows: {df.count()}, Total columns: {len(df.columns)}")

# -------------------- Categorical Encoding --------------------
print("[STATUS] Identifying categorical columns...")
categorical_columns = [col_name for col_name, dtype in df.dtypes if dtype == 'string']
print(f"[STATUS] Categorical columns found: {categorical_columns}")

for column in categorical_columns:
    unique_values = df.select(column).distinct().rdd.flatMap(lambda x: x).collect()
    unique_values = sorted(unique_values)
    mapping_dict = {val: idx for idx, val in enumerate(unique_values)}
    print(f"[STATUS] Encoding '{column}' with mapping: {mapping_dict}")

    def map_value(val):
        return mapping_dict.get(val, -1)
    map_udf = udf(map_value, IntegerType())
    new_col_name = column + '_indexed'
    df = df.withColumn(new_col_name, map_udf(df[column]))

print("[STATUS] Sample of encoded categorical columns:")
df.select(*categorical_columns, *[c + '_indexed' for c in categorical_columns]).show(5)

# Drop original categorical columns
df_final = df.drop(*categorical_columns)

# -------------------- Data Type Conversion --------------------
print("[STATUS] Casting all columns to float...")
for col_name in df_final.columns:
    df_final = df_final.withColumn(col_name, col(col_name).cast('float'))
df_final.printSchema()

# -------------------- Feature Preparation --------------------
energy_column = 'Energy_Consumption'
feature_columns = [col_name for col_name in df_final.columns if col_name != energy_column]

print("[STATUS] Preparing features and labels...")
features = df_final.select(*feature_columns).rdd.map(lambda row: [float(x) for x in row]).collect()
labels = df_final.select(energy_column).rdd.map(lambda row: float(row[0])).collect()

labeled_points = [LabeledPoint(label, feature) for label, feature in zip(labels, features)]
data_rdd = sc.parallelize(labeled_points)

# -------------------- Train/Test Split --------------------
print("[STATUS] Splitting data into training and test sets...")
train_rdd, test_rdd = data_rdd.randomSplit([0.70, 0.30], seed=21)

# -------------------- Model Training --------------------
print("[STATUS] Training Gradient Boosted Trees regressor...")
model = GradientBoostedTrees.trainRegressor(
    train_rdd,
    categoricalFeaturesInfo={},
    numIterations=125,
    learningRate=0.2,
    maxDepth=2
)

# -------------------- Prediction and Evaluation --------------------
print("[STATUS] Generating predictions and evaluating model...")
predictions = model.predict(test_rdd.map(lambda x: x.features))
predictionAndLabel = predictions.zip(test_rdd.map(lambda x: x.label))
metrics = RegressionMetrics(predictionAndLabel)

print("\n=============== Model Performance ===============")
print(f"Root Mean Squared Error (RMSE): {metrics.rootMeanSquaredError:.4f}")
print(f"R2 (coefficient of determination): {metrics.r2:.4f}")
print("=================================================\n")

# -------------------- Model Saving --------------------
print("[STATUS] Saving the trained model to 'reg' directory...")
model.save(sc, 'regression')
print("[STATUS] Model saved successfully!\n")
