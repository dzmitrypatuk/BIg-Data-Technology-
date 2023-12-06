# Databricks notebook source
# MAGIC %pip install "boto3>=1.28" "s3fs>=2023.3.0"
# MAGIC %pip install pyarrow fastparquet
# MAGIC %pip install seaborn
# MAGIC
# MAGIC import boto3
# MAGIC import time
# MAGIC import numpy as np
# MAGIC import pandas as pd 
# MAGIC import os
# MAGIC import seaborn as sns
# MAGIC import pyarrow
# MAGIC #import fastparquet
# MAGIC
# MAGIC pd.options.display.float_format = '{:.4f}'.format
# MAGIC
# MAGIC sc.version

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

def perform_EDA(df, filename):
    # print(df.head(5))
    print(f"{filename} Number of records:{len(df)}" )
    print(f"{filename} Info")
    print(df.info())
    print(f"{filename} Describe")
    print(df.describe())
    print(f"{filename} Columns with null values")
    print(df.columns[df.isnull().any()].tolist())
    rows_with_null_values = df.isnull().any(axis=1).sum()
    print(f"{filename} Number of Rows with null values: {rows_with_null_values}" )
    integer_column_list = df.select_dtypes(include='int64').columns
    print(f"{filename} Integer data type columns: {integer_column_list}")
    float_column_list = df.select_dtypes(include='float64').columns
    print(f"{filename} Float data type columns: {float_column_list}")
    # Add other codes here to explore and visualize specific columns   

# COMMAND ----------

# Amazon Product Reviews
# Path to Amazon S3 files
filepath = "s3://amazon-reviews-project-dp/landing/"
# List of data files
filename_list = ['amazon_reviews_us_Apparel_v1_00.tsv'] 

for filename in filename_list:
    # Read in amazon reviews. Reminder: Tab-separated values files
    print(f"Working on file: {filename}")
    reviews_df = pd.read_csv(f"{filepath}{filename}", sep='\t', on_bad_lines='skip', low_memory=False)
    reviews_df.head()
    perform_EDA(reviews_df,filename)

# COMMAND ----------

#sdf1 = spark.read.json('https://amazon-reviews-project-dp.s3.us-east-2.amazonaws.com/landing/amazon_reviews_us_Digital_Software_v1_00.tsv', sep='\t')

# COMMAND ----------

print(loaded_data_pd_df1)

# COMMAND ----------

reviews_sdf.head()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

#Data 

# COMMAND ----------


