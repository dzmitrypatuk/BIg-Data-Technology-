# Databricks notebook source
# MAGIC %pip install "boto3>=1.28" "s3fs>=2023.3.0"
# MAGIC %pip install -U textblob
# MAGIC #%%python -m textblob.download_corpora
# MAGIC
# MAGIC import boto3
# MAGIC import time
# MAGIC import os
# MAGIC import seaborn as sns
# MAGIC import matplotlib.pyplot as plt
# MAGIC import numpy as np
# MAGIC import pandas as pd 
# MAGIC
# MAGIC import pyarrow
# MAGIC
# MAGIC from pyspark.sql.functions import col, isnan, isnull, when, count, udf, size, split, year, month, format_number, date_format, length
# MAGIC from pyspark.sql.types import IntegerType, DateType, StringType, StructType, DoubleType
# MAGIC from pyspark.sql.functions import *
# MAGIC from pyspark.ml.feature import FeatureHasher
# MAGIC from pyspark.ml.feature import MinMaxScaler
# MAGIC from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Normalizer, StandardScaler, HashingTF, IDF, Tokenizer, RegexTokenizer
# MAGIC
# MAGIC from pyspark.ml import Pipeline
# MAGIC from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
# MAGIC from pyspark.ml.regression import GeneralizedLinearRegression
# MAGIC from pyspark.ml.evaluation import BinaryClassificationEvaluator
# MAGIC from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# MAGIC
# MAGIC from pyspark.ml.linalg import DenseMatrix, Vectors
# MAGIC from pyspark.ml.stat import Correlation
# MAGIC
# MAGIC #from textblob import TextBlob
# MAGIC from pyspark.ml import Pipeline
# MAGIC
# MAGIC # Import the logistic regression model
# MAGIC # Import the evaluation module
# MAGIC from pyspark.ml.evaluation import *
# MAGIC # Import the model tuning module
# MAGIC from pyspark.ml.tuning import *
# MAGIC

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

# Amazon Product Reviews
# Path to Amazon S3 files
filepath = "s3://amazon-reviews-project-dp/raw/"
out_filepath = "s3://amazon-reviews-project-dp/trusted/"

# List of data files
filename_list = ['cleaned_amazon_reviews_us_Apparel_v1_00.tsv.parquet']

#Read files from the list of files names in the s3 bucket
for filename in filename_list:
    # Read in amazon reviews. Reminder: Tab-separated values files
    print(f"Working on file: {filename}")
    reviews_sdf = spark.read.parquet(f"{filepath}{filename}")
    reviews_sdf.printSchema()
    reviews_sdf.show()


# COMMAND ----------

selected_sdf = reviews_sdf.select(['star_rating', 'total_votes', 'review_body', 'vine', 'product_category', 'verified_purchase', 'helpful_votes', 'clean_review_headline', 'clean_review_body']).sample(False, 0.1, 42)


# Create a count of the review words
selected_sdf = selected_sdf.withColumn('review_body_wordcount', size(split(col('review_body'), ' ')))

selected_sdf = selected_sdf.withColumn("total_votes",selected_sdf.total_votes.cast(DoubleType()))
selected_sdf = selected_sdf.withColumn("helpful_votes",selected_sdf.total_votes.cast(DoubleType()))
selected_sdf = selected_sdf.withColumn("review_body_wordcount",selected_sdf.review_body_wordcount.cast(DoubleType()))

# Create a label. =1 if over 3, =0 if otherwise
selected_sdf = selected_sdf.withColumn("label", when(selected_sdf.star_rating >= 4, 1.0).otherwise(0.0) )

# tokenizer = Tokenizer(inputCol="clean_review_body", outputCol="clean_review_words")
tokenizer = RegexTokenizer(inputCol="clean_review_body", outputCol="clean_review_words", pattern="\\w+", gaps=False)
selected_sdf = tokenizer.transform(selected_sdf)

# Run the hash function over the tokens
selected_sdf = selected_sdf.drop('clean_review_tf')
hashtf = HashingTF(numFeatures=2**13, inputCol="clean_review_words", outputCol='clean_review_tf')
selected_sdf = hashtf.transform(selected_sdf)

idf = IDF(inputCol='clean_review_tf', outputCol="clean_review_features", minDocFreq=5)
selected_sdf = idf.fit(selected_sdf).transform(selected_sdf)

# Create an indexer for the string based columns.
indexer = StringIndexer(inputCols=["product_category", "vine", "verified_purchase"], outputCols=["product_categoryIndex", "vineIndex", "verified_purchaseIndex"], handleInvalid="keep")

# Create an encoder for the indexes
encoder = OneHotEncoder(inputCols=["product_categoryIndex", "vineIndex", "verified_purchaseIndex" ],
                        outputCols=["product_categoryVector", "vineVector", "verified_purchaseVector" ], dropLast=True, handleInvalid="keep")

# Assemble all of the vectors together into one large vector named "features"
# Inlcude the vector from the TF/IDF "clean_review_features"
assembler = VectorAssembler(inputCols=["product_categoryVector", "vineVector", "verified_purchaseVector", "total_votes", "review_body_wordcount", "clean_review_features"], outputCol="features")

# Build the pipeline with all of the stages for indexer, encoder, toekenizer etc.
reviews_pipe = Pipeline(stages=[indexer, encoder, assembler])

# Call .fit to transform the data
transformed_sdf = reviews_pipe.fit(selected_sdf).transform(selected_sdf)

transformed_sdf.show()

# COMMAND ----------

# Split the data into 70% training and 30% test sets
trainingData, testData = transformed_sdf.randomSplit([0.7, 0.3], seed=42)
# Create a LogisticRegression Estimator
lr = LogisticRegression()
# Fit the model to the training data
model = lr.fit(trainingData)
# Show model coefficients and intercept
print("Coefficients: ", model.coefficients)
print("Intercept: ", model.intercept)
# Test the model on the testData
test_results = model.transform(testData)
# Show the test results
test_results.select('product_category', 'vine', 'verified_purchase', 'clean_review_body', 'rawPrediction','probability','prediction',
'label').show(truncate=False)


# COMMAND ----------

# Save the confusion matrix
cm = test_results.groupby('label').pivot('prediction').count().fillna(0).collect()
def calculate_recall_precision(cm):
    tn = cm[0][1] # True Negative
    fp = cm[0][2] # False Positive
    fn = cm[1][1] # False Negative
    tp = cm[1][2] # True Positive
    precision = tp / ( tp + fp )
    recall = tp / ( tp + fn )
    accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
    f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) )
    return accuracy, precision, recall, f1_score
print( calculate_recall_precision(cm))

# COMMAND ----------

# Create a BinaryClassificationEvaluator to evaluate how well the model works
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
# Create the parameter grid (empty for now)
grid = ParamGridBuilder().build()
# Create the CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3 )
# Use the CrossValidator to Fit the training data
cv = cv.fit(trainingData)
# Show the average performance over the three folds
cv.avgMetrics
# Evaluate the test data using the cross-validator model
# Reminder: We used Area Under the Curve
evaluator.evaluate(cv.transform(testData))

# COMMAND ----------

cv.avgMetrics


# COMMAND ----------

# Test the model on the testData
test_results = model.transform(testData)
# Confusion matrix on the test_results for Logistic regression
test_results.groupby('label').pivot('prediction').count().show()

# COMMAND ----------

# Create a grid to hold hyperparameters
grid = ParamGridBuilder()
grid = grid.addGrid(lr.regParam, [0.2, 0.4, 0.6, 0.8] )
grid = grid.addGrid(lr.elasticNetParam, [0, 1])
# Build the grid
grid = grid.build()
print('Number of models to be tested: ', len(grid))
# Create the CrossValidator using the new hyperparameter grid
cv1 = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
# Call cv.fit() to create models with all of the combinations of parameters in the grid
all_models = cv1.fit(trainingData)
print("Average Metrics for Each model: ", all_models.avgMetrics)


# COMMAND ----------

# Gather the metrics and parameters of the model with the best average metrics
hyperparams = all_models.getEstimatorParamMaps()[np.argmax(all_models.avgMetrics)]
# Print out the list of hyperparameters for the best model
for i in range(len(hyperparams.items())):
    print([x for x in hyperparams.items()][i])
#(Param(parent='LogisticRegression_2effdf339a6c', name='regParam', doc='regularization parameter (>= 0).'), 0.4)
#(Param(parent='LogisticRegression_2effdf339a6c', name='elasticNetParam', doc='the ElasticNet mixing parameter, in
#range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'), 0.0)
# Choose the best model
bestModel = all_models.bestModel
print("Area under ROC curve:", bestModel.summary.areaUnderROC)
# Area under ROC curve: 1.0

# COMMAND ----------

# Use the model 'bestModel' to predict the test set
test_results = bestModel.transform(testData)
# Show the results
test_results.select('product_category', 'vine', 'verified_purchase', 'clean_review_body', 'rawPrediction','probability','prediction',
'label').show(truncate=False)
# Evaluate the predictions. Area Under ROC curve
print(evaluator.evaluate(test_results))

# COMMAND ----------


# Save the trusted dataframe as Parquet
output_file_path= f"{out_filepath}trusted_{filename}"
#output_file_path= f"{filepath}{filename}"
transformed_sdf.write.parquet(output_file_path) 

# COMMAND ----------

#model path
model_path = "s3://amazon-reviews-project-dp/models/"

# Save the best model
model_name = "amazon_reviews_logistic_regression_model"
model.write().overwrite().save(f"{model_path}{model_name}")

# COMMAND ----------

### Code parts 
    # Filter out short review body texts
    #selected_sdf = selected_sdf.where(length(selected_sdf.clean_review_body) > 10)

    # Filter out review body texts with 5 or fewer words
    #selected_sdf = selected_sdf.where(selected_sdf.review_body_wordcount > 5)
