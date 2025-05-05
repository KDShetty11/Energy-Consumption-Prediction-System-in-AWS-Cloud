import sys
import warnings
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from pyspark.mllib.tree import GradientBoostedTreesModel
from pyspark.mllib.evaluation import RegressionMetrics

warnings.filterwarnings("ignore")

def main(test_file_path):
    # Spark setup
    conf = SparkConf().setAppName('EnergyConsumptionPrediction').setMaster('local')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # Load test CSV file
    df = spark.read.csv(test_file_path, header=True, inferSchema=True)

    # 1. Rename columns (remove spaces)
    for column in df.columns:
        df = df.withColumnRenamed(column, column.replace(' ', '_'))

    # 2. Identify categorical columns
    categorical_columns = [col_name for col_name, dtype in df.dtypes if dtype == 'string']
        
    print(categorical_columns)

    # 3. Encode categorical columns
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

    # 4. Drop original categorical columns
    df_final = df.drop(*categorical_columns)

    # 5. Cast all columns to float
    for col_name in df_final.columns:
        df_final = df_final.withColumn(col_name, col(col_name).cast('float'))

    # 6. Prepare features list
    energy_column = 'Energy_Consumption'
    feature_columns = [col_name for col_name in df_final.columns if col_name != energy_column]

    features_rdd = df_final.select(*feature_columns).rdd.map(lambda row: [float(x) for x in row])
    labels_rdd = df_final.select(energy_column).rdd.map(lambda row: float(row[0]))

    # 7. Load trained model
    model = GradientBoostedTreesModel.load(sc, 's3://energypreddatasets/model/regression_model')

    # 8. Predict
    predictions = model.predict(features_rdd)

    # 9. Pair predictions with actual labels
    predictionAndLabel = predictions.zip(labels_rdd)

    # 10. Evaluate using RegressionMetrics
    metrics = RegressionMetrics(predictionAndLabel)
    

    # 11. Output predictions
    #predictions_list = predictions.collect()

    # print("-----------Predictions-----------")
    # for idx, pred in enumerate(predictions_list):
    #     print(f"Sample {idx + 1}: Predicted Energy Consumption = {pred:.2f}")

    # 12. Output evaluation
    print("---------------Evaluation Metrics-----------------")
    print(f"Root Mean Squared Error (RMSE): {metrics.rootMeanSquaredError}")
    print(f"R2 (coefficient of determination): {metrics.r2}")

    # 13. Clean up
    sc.stop()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: spark-submit predict.py <path_to_test_csv>")
        sys.exit(1)

    test_file_path = sys.argv[1]
    main(test_file_path)
