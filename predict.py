import sys
import warnings
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.mllib.tree import GradientBoostedTreesModel
from pyspark.mllib.evaluation import RegressionMetrics

warnings.filterwarnings("ignore")

def print_progress(step):
    """Print a clean progress message for a given step."""
    print(f"[✓] {step:<30}")

def print_metrics(rmse, r2):
    """Print evaluation metrics in a styled table format."""
    print("\n" + "=" * 40)
    print("       Evaluation Metrics       ")
    print("=" * 40)
    print(f"{'Metric':<30} | {'Value':>8}")
    print("-" * 40)
    print(f"{'Root Mean Squared Error (RMSE)':<30} | {rmse:>8.4f}")
    print(f"{'R2 (Coefficient of Determination)':<30} | {r2:>8.4f}")
    print("=" * 40 + "\n")

def main(test_file_path):
    # Initialize SparkSession
    print_progress("Initializing SparkSession")
    spark = SparkSession.builder \
        .appName('EnergyConsumptionPrediction') \
        .master('local') \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        # Load test CSV file
        print_progress("Loading test CSV")
        df = spark.read.csv(test_file_path, header=True, inferSchema=True).cache()
        
        # Rename columns to remove spaces
        print_progress("Renaming columns")
        for column in df.columns:
            df = df.withColumnRenamed(column, column.replace(' ', '_'))

        # Identify and encode categorical columns
        print_progress("Encoding categorical columns")
        categorical_columns = [col_name for col_name, dtype in df.dtypes if dtype == 'string']
        for column in categorical_columns:
            unique_values = df.select(column).distinct().rdd.flatMap(lambda x: x).collect()
            unique_values = sorted(unique_values)
            mapping_dict = {val: idx for idx, val in enumerate(unique_values)}

            def map_value(val):
                return mapping_dict.get(val, -1)

            map_udf = udf(map_value, IntegerType())
            df = df.withColumn(column + '_indexed', map_udf(df[column]))

        # Drop original categorical columns
        print_progress("Dropping categorical columns")
        df_final = df.drop(*categorical_columns)

        # Cast all columns to float
        print_progress("Casting columns to float")
        for col_name in df_final.columns:
            df_final = df_final.withColumn(col_name, col(col_name).cast('float'))

        # Prepare features and labels
        print_progress("Preparing features and labels")
        energy_column = 'Energy_Consumption'
        feature_columns = [col_name for col_name in df_final.columns if col_name != energy_column]

        # Convert to RDD for model prediction
        print_progress("Converting to RDD")
        features_rdd = df_final.select(*feature_columns).rdd.map(lambda row: [float(x) for x in row])
        labels_rdd = df_final.select(energy_column).rdd.map(lambda row: float(row[0]))

        # Load trained model
        print_progress("Loading trained model")
        model = GradientBoostedTreesModel.load(spark.sparkContext, '/content/regression_model')

        # Predict
        print_progress("Making predictions")
        predictions = model.predict(features_rdd)

        # Pair predictions with actual labels
        print_progress("Pairing predictions with labels")
        prediction_and_label = predictions.zip(labels_rdd)

        # Evaluate using RegressionMetrics
        print_progress("Evaluating model")
        metrics = RegressionMetrics(prediction_and_label)

        # Output evaluation metrics
        print_metrics(metrics.rootMeanSquaredError, metrics.r2)

    finally:
        # Clean up
        print_progress("Stopping SparkSession")
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <test_dataset_path>")
        sys.exit(1)
    
    test_file_path = sys.argv[1]
    main(test_file_path)