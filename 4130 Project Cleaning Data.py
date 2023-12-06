# Databricks notebook source
#%pip install "boto3>=1.28" "s3fs>=2023.3.0"

import boto3
import time
import os
import seaborn as sns
import pyarrow

from pyspark.sql.functions import col, isnan, when, count, udf

# COMMAND ----------

# To work with Amazon S3 storage, set the following variables using your AWS Access Key and Secret Key
# Set the Region to where your files are stored in S3.
access_key = ''
secret_key = ''
# Set the environment variables so boto3 can pick them up later
os.environ['AWS_ACCESS_KEY_ID'] = access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
encoded_secret_key = secret_key.replace("/", "%2F").replace("+", "%2B")
aws_region = "us-east-2"
# Set this to the name of your bucket where the files are stored
aws_bucket_name = "amazon-reviews-project-dp"
mount_name = "s3dataread"

# COMMAND ----------

sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key) 

sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key) 

sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3." + aws_region + ".amazonaws.com")

# COMMAND ----------

# The Amazon S3 bucket can be mounted like a local directory using the Databricks dbfs file system
# Mount Amazon S3 Bucket as a local file system
#dbutils.fs.mount(f"s3a://{access_key}:{encoded_secret_key}@{aws_bucket_name}", f"/mnt/{mount_name}")
#display(dbutils.fs.ls(f"/mnt/{mount_name}"))
#file_location = "dbfs:/mnt/s3dataread/landing/*"

# COMMAND ----------

#function to clean data and prepare dataframe for the 'raw' folder  
def perform_EDA(sdf, filename):
    print(f"{filename} Number of records:{sdf.count()}" )
    print('Number of records: ', reviews_sdf.count())
    reviews_sdf.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in ["star_rating", "review_body"]] ).show()

def perform_cleaning(sdf, filename, out_filepath):
    sdf.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in ["star_rating", "review_body"]] ).show()
    sdf = sdf.na.drop(subset=["star_rating", "review_body"])
    # Turn this function into a User-Defined Function (UDF)
    ascii_udf = udf(ascii_only)
    # Clean up the review_headline and review_body
    sdf = sdf.withColumn("clean_review_headline", ascii_udf('review_headline'))
    sdf = sdf.withColumn("clean_review_body", ascii_udf('review_body'))
    # Re-check the cleaned headline and body
    sdf.select("clean_review_headline", "clean_review_body").summary("count", "min", "max").show()
    sdf.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in ["star_rating", "review_body"]] ).show()

    # Save the cleaned dataframe as Parquet
    #output_file_path="s3://amazon-reviews-project-dp/raw/cleaned_amazon_reviews_us_Apparel_v1_00.parquet"
    output_file_path= f"{out_filepath}cleaned_{filename}.parquet"
    #output_file_path= f"{filepath}{filename}"
    sdf.write.parquet(output_file_path) 


# Define a function to strip out any non-ascii character
def ascii_only(mystring):
    if mystring:
        return mystring.encode('ascii', 'ignore').decode('ascii')
    else:
        return None

# COMMAND ----------

# Amazon Product Reviews
# Path to Amazon S3 files
filepath = "s3://amazon-reviews-project-dp/landing/"
out_filepath = "s3://amazon-reviews-project-dp/raw/"

# List of data files
filename_list = [ 'amazon_reviews_us_Apparel_v1_00.tsv']

#Read files from the list of files names in the s3 bucket
for filename in filename_list:
    # Read in amazon reviews. Reminder: Tab-separated values files
    print(f"Working on file: {filename}")
    reviews_sdf = spark.read.csv(f"{filepath}{filename}", sep='\t', header=True, inferSchema=True)
    reviews_sdf.printSchema()
    #perform_EDA(reviews_df,filename)
    perform_cleaning(reviews_sdf, filename, out_filepath)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


