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
promotion_dates_path = "dbfs:/user/hive/warehouse/promotion_dates"

# Step 1: Data Ingestion
# Since the data is stored in Delta Lake format, use 'format("delta")' to read it
sales_sdf = spark.read.format("delta").load(sales_data_path)
promotions_sdf = spark.read.format("delta").load(promotion_dates_path)

# Convert Spark DataFrames to Pandas on Spark DataFrames
sales_pdf = sales_sdf.pandas_api()
promotions_pdf = promotions_sdf.pandas_api()


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, weekofyear, sum as spark_sum, udf
from pyspark.sql.types import DateType
import datetime
from pyspark.sql.functions import lit

from pyspark.sql.functions import explode, sequence, dayofweek
from pyspark.sql.functions import explode, expr

# Initialize the Spark session
spark = SparkSession.builder.appName("PromotionEffectAnalysis").getOrCreate()

# Define the paths to the Delta tables
sales_data_path = "dbfs:/user/hive/warehouse/assignment_4_1_a"
promotion_dates_path = "dbfs:/user/hive/warehouse/promotion_dates"

# Step 1: Data Ingestion
# Since the data is stored in Delta Lake format, use 'format("delta")' to read it
sales_sdf = spark.read.format("delta").load(sales_data_path)
promotions_sdf = spark.read.format("delta").load(promotion_dates_path)

# Step 2: Data Preparation

# Define UDF for parsing dates
def parse_date(date_str):
    for fmt in ("%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None

parse_date_udf = udf(parse_date, DateType())

# Apply UDF to parse dates in promotions DataFrame
promotions_sdf = promotions_sdf.withColumn("StartDate", parse_date_udf(col("StartDate"))) \
                               .withColumn("EndDate", parse_date_udf(col("EndDate")))


# Manually correct the StartDate for Promo1
promotions_sdf = promotions_sdf.withColumn("StartDate", 
                                           when(col("Period") == "Promo1", 
                                                to_date(lit("2015-02-10"), "yyyy-MM-dd")).otherwise(col("StartDate")))

# Clean the sales data to handle negative sales quantities
sales_sdf = sales_sdf.withColumn("SalesQuantity", when(col("SalesQuantity") < 0, 0).otherwise(col("SalesQuantity")))

# Add week of year column
sales_sdf = sales_sdf.withColumn("WeekOfYear", weekofyear(to_date(col("Date"), 'yyyy/MM/dd')))


# COMMAND ----------

promotions_sdf.show()

# COMMAND ----------

# Convert Spark DataFrames to Pandas on Spark DataFrames
sales_pdf = sales_sdf.pandas_api()
promotions_df = promotions_sdf.toPandas()

import pandas as pd
# Create a DataFrame with a range of dates for each promotion
date_ranges = promotions_df.apply(lambda row: pd.date_range(start=row['StartDate'], end=row['EndDate']), axis=1)
# Flatten the list of date ranges and associate them with the corresponding promotion
flat_list = [(row['Period'], date) for _, row in promotions_pdf.iterrows() for date in date_ranges[_]]
promotions_df = pd.DataFrame(flat_list, columns=['PromoId', 'Date'])

# Extract week of year
promotions_df['WeekOfYear'] = promotions_df['Date'].dt.isocalendar().week

# Resulting DataFrame
promotions_sdf = spark.createDataFrame(promotions_df) 

# COMMAND ----------


