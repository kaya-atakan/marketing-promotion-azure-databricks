# Databricks notebook source
# Import the necessary PySpark functions
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, weekofyear, avg

# Initialize the Spark session
spark = SparkSession.builder.appName("PromotionEffectAnalysis").getOrCreate()

# Step 1: Data Ingestion
# Since the data is stored in Delta Lake format, use 'format("delta")' to read it
sales_df = spark.read.format("delta").load("dbfs:/user/hive/warehouse/assignment_4_1_a")
promotions_df = spark.read.format("delta").load("dbfs:/user/hive/warehouse/promotion_dates")


# COMMAND ----------


