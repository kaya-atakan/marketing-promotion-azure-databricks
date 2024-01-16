# Databricks notebook source
# Import the necessary PySpark functions
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, weekofyear, avg
import pyspark.pandas as ps

# Initialize the Spark session
spark = SparkSession.builder.appName("PromotionEffectAnalysis").getOrCreate()

# Set the Pandas on Spark default index type for efficiency
ps.set_option('compute.default_index_type', 'distributed-sequence')

# Define the paths to the Delta tables
sales_data_path = "dbfs:/user/hive/warehouse/assignment_4_1_a"
promotion_dates_path = "dbfs:/user/hive/warehouse/promotiondates"

# Step 1: Data Ingestion
# Since the data is stored in Delta Lake format, use 'format("delta")' to read it
sales_sdf = spark.read.format("delta").load(sales_data_path)
promotions_sdf = spark.read.format("delta").load(promotion_dates_path)

# Convert Spark DataFrames to Pandas on Spark DataFrames
sales_pdf = sales_sdf.to_pandas_on_spark()
promotions_pdf = promotions_sdf.to_pandas_on_spark()


# COMMAND ----------


