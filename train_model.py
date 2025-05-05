import numpy as np
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.evaluation import RegressionMetrics
import warnings

warnings.filterwarnings("ignore")

conf = SparkConf().setAppName('EnergyConsumptionPrediction').setMaster('local')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

df = spark.read.csv("s3://energypreddatasets/data/TrainingDataset.csv", header=True, inferSchema=True)
for column in df.columns:
    df = df.withColumnRenamed(column, column.replace(' ', '_'))
df.printSchema()
print(df.count())
print(len(df.columns))

categorical_columns = [col_name for col_name, dtype in df.dtypes if dtype == 'string']

print(categorical_columns)

for column in categorical_columns:
    unique_values = df.select(column).distinct().rdd.flatMap(lambda x: x).collect()
    unique_values = sorted(unique_values) 
    mapping_dict = {val: idx for idx, val in enumerate(unique_values)}
    print(f"Mapping for {column}:", mapping_dict)

    def map_value(val):
        return mapping_dict.get(val, -1) 
    
    map_udf = udf(map_value, IntegerType())

    new_col_name = column + '_indexed'
    df = df.withColumn(new_col_name, map_udf(df[column]))

df.select(*categorical_columns, *[c + '_indexed' for c in categorical_columns]).show(5)

df_final = df.drop(*categorical_columns)

for col_name in df_final.columns:
  df_final = df_final.withColumn(col_name, col(col_name).cast('float'))
df_final.printSchema()

energy_column = 'Energy_Consumption'

feature_columns = [col_name for col_name in df_final.columns if col_name != energy_column] 

features = df_final.select(*feature_columns).rdd.map(lambda row: [float(x) for x in row]).collect()
labels = df_final.select(energy_column).rdd.map(lambda row: float(row[0])).collect()

labeled_points = [LabeledPoint(label, feature) for label, feature in zip(labels, features)]
data_rdd = sc.parallelize(labeled_points)

train_rdd, test_rdd = data_rdd.randomSplit([0.7, 0.3], seed=21)

model = GradientBoostedTrees.trainRegressor(
    train_rdd,
    categoricalFeaturesInfo={}, 
    numIterations=125,  
    learningRate=0.2,                
    maxDepth=2                  
)

predictions = model.predict(test_rdd.map(lambda x: x.features))

predictionAndLabel = predictions.zip(test_rdd.map(lambda x: x.label))

metrics = RegressionMetrics(predictionAndLabel)

print("---------------Output-----------------")
print(f"Root Mean Squared Error (RMSE): {metrics.rootMeanSquaredError}")
print(f"R2 (coefficient of determination): {metrics.r2}")

model.save(sc, 'regression_model')