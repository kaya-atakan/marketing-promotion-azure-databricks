# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, to_date, col, weekofyear, sum as spark_sum, udf
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

sales_sdf.display(10)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, IntegerType, ArrayType, StringType
from pyspark.sql.functions import col, udf, sum, weekofyear, size
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler


# Filter for the first 4 promotions
promo_data = promotions_sdf.filter(promotions_sdf["Period"].isin(["Promo1", "Promo2", "Promo3", "Promo4"]))

# Function to determine if a date is within the promotion period
def is_promotion(date, promo_periods):
    for start, end in promo_periods:
        if start <= date <= end:
            return 1
    return 0

# Collect promotion dates into a list for broadcasting
promo_periods = promo_data.select("StartDate", "EndDate").collect()
promo_periods = [(row.StartDate, row.EndDate) for row in promo_periods]

# Broadcast the promo_periods list
broadcasted_promo_periods = spark.sparkContext.broadcast(promo_periods)

# Define a UDF to mark promotion periods in sales data
is_promo_udf = udf(lambda date: is_promotion(date, broadcasted_promo_periods.value), IntegerType())

# Add a column to mark if a sale is during a promotion
sales_sdf = sales_sdf.withColumn("is_promo", is_promo_udf(col("Date")))
sales_sdf = sales_sdf.filter(col("Date").isNotNull())

# Handling negative sales values (returns)
sales_sdf = sales_sdf.filter(sales_sdf["SalesQuantity"] >= 0)

# COMMAND ----------

sales_sdf.show(10)

# COMMAND ----------

sales_sdf.display(10)

# COMMAND ----------

from pyspark.sql.functions import col, when, lit

# Function to determine the promotion period category
def determine_promotion_period(date, promo_periods):
    for period, start, end in promo_periods:
        if start <= date <= end:
            return f"During-{period}"
    return "Outside-Promotion"

# Broadcast the promo_periods list with promotion IDs
promo_periods_with_id = [(row.Period, row.StartDate, row.EndDate) for row in promo_data.collect()]
broadcasted_promo_periods_with_id = spark.sparkContext.broadcast(promo_periods_with_id)

# Define a UDF to determine promotion period category
determine_promotion_period_udf = udf(lambda date: determine_promotion_period(date, broadcasted_promo_periods_with_id.value), StringType())

# Add a column to mark the promotion period category
sales_sdf = sales_sdf.withColumn("PromotionPeriodCategory", determine_promotion_period_udf(col("Date")))


# COMMAND ----------

sales_sdf.display(10)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.sql.functions import dayofweek, month


# Feature engineering: index the categorical column and assemble features
indexer = StringIndexer(inputCol="PromotionPeriodCategory", outputCol="PromotionPeriodCategoryIndex")


# Extract additional features from the 'Date' column
sales_sdf = sales_sdf.withColumn("DayOfWeek", dayofweek(col("Date")))
sales_sdf = sales_sdf.withColumn("Month", month(col("Date")))
sales_sdf = sales_sdf.drop("is_promo")

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

# Feature engineering: index the categorical column and assemble features
indexer = StringIndexer(inputCol="PromotionPeriodCategory", outputCol="PromotionPeriodCategoryIndex")
assembler = VectorAssembler(
    inputCols=["PromotionPeriodCategoryIndex", "StoreCode", "ProductCode", "WeekOfYear", "DayOfWeek", "Month"],
    outputCol="features"
)

# Define the model
rf = RandomForestRegressor(featuresCol="features", labelCol="SalesQuantity")

# Define the pipeline
pipeline = Pipeline(stages=[indexer, assembler, rf])

# Train the model on the entire dataset
model = pipeline.fit(sales_sdf)

# Since there's no test set, we can't evaluate the model in the usual way.
# However, you can still use the model to make predictions on new data.



# COMMAND ----------

# Load the new data
new_sales_data_path = "dbfs:/user/hive/warehouse/assignment_4_1_b"

new_sales_sdf = spark.read.format("delta").load(new_sales_data_path)

# Apply the same preprocessing steps (you'll need to adjust these steps based on your initial preprocessing)
new_sales_sdf = new_sales_sdf.withColumn("SalesQuantity", when(col("SalesQuantity") < 0, 0).otherwise(col("SalesQuantity")))
new_sales_sdf = new_sales_sdf.withColumn("WeekOfYear", weekofyear(to_date(col("Date"), 'yyyy/MM/dd')))
new_sales_sdf = new_sales_sdf.withColumn("DayOfWeek", dayofweek(col("Date")))
new_sales_sdf = new_sales_sdf.withColumn("Month", month(col("Date")))

# Add the 'PromotionPeriodCategory' column based on Promotion 5 dates
# Assume we know the StartDate and EndDate of Promotion 5
promo5_start_date = "2015-09-01"
promo5_end_date = "2015-09-06"
new_sales_sdf = new_sales_sdf.withColumn(
    "PromotionPeriodCategory",
    when(
        col("Date").between(lit(promo5_start_date), lit(promo5_end_date)),
        "During-Promotion"
    ).otherwise("Outside-Promotion")
)

# COMMAND ----------

new_sales_sdf.display(10)

# COMMAND ----------

# Assuming 'model' is the trained PipelineModel
string_indexer_model = model.stages[0]  # Replace 0 with the correct index of StringIndexerModel in your pipeline
vector_assembler = model.stages[1]  # Replace 1 with the correct index of VectorAssembler in your pipeline

# Transform the new data with StringIndexerModel
# new_sales_sdf = string_indexer_model.transform(new_sales_sdf)

# Now apply VectorAssembler
# new_sales_sdf = vector_assembler.transform(new_sales_sdf)

# COMMAND ----------

# Drop existing columns that will be recreated
new_sales_sdf = new_sales_sdf.drop('PromotionPeriodCategoryIndex', 'features')

# Apply StringIndexerModel transformation
new_sales_sdf = string_indexer_model.transform(new_sales_sdf)

# Apply VectorAssembler transformation
new_sales_sdf = vector_assembler.transform(new_sales_sdf)

# Assuming 'model' is the RandomForestRegressor model extracted from your trained pipeline
# Now make predictions

# Extract the RandomForestRegressor model from the trained pipeline
rf_model = model.stages[-1]  # Assuming RandomForestRegressor is the last stage in your pipeline

new_predictions = rf_model.transform(new_sales_sdf)



# COMMAND ----------

new_predictions.display(10)

# COMMAND ----------


