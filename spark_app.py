import pyspark
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, count
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Set these paths according to your installation
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk1.8.0_202"
os.environ["SPARK_HOME"] = r"C:\spark"

# Add Spark python files to PATH so PySpark can find them

sys.path.append(os.path.join(os.environ["SPARK_HOME"], 'python'))
sys.path.append(os.path.join(os.environ["SPARK_HOME"], 'python', 'lib', 'py4j-src.zip'))

# Setup Spark session
spark = SparkSession.builder \
    .appName("Twitter Sentiment Analysis") \
    .getOrCreate()

# Load CSV dataset
df = spark.read.csv('dataset/twitter_sentiment.csv', header=True, inferSchema=True)

# Data Preprocessing
df_clean = df.withColumn("clean_text",
    lower(
        regexp_replace(
            regexp_replace(
                regexp_replace(col("text"), r"http\\S+", ""),
                r"@[A-Za-z0-9]+", ""
            ),
            r"#[A-Za-z0-9]+", ""
        )
    )
)

# Remove non-alphabetic and extra whitespace from clean_text
df_clean = df_clean.withColumn("clean_text",
    regexp_replace(col("clean_text"), r"[^a-zA-Z\\s]", "")
)
df_clean = df_clean.withColumn("clean_text",
    regexp_replace(col("clean_text"), r"\\s+", " ")
)

# Sentiment Distribution
sentiment_counts = df_clean.groupBy("airline_sentiment").count().orderBy(col("count").desc())
sentiment_pd = sentiment_counts.toPandas()
plt.figure(figsize=(6,4))
sns.barplot(x='airline_sentiment', y='count', data=sentiment_pd)
plt.title('Overall Sentiment Distribution')
plt.savefig('dataset/sentiment_distribution.png')
plt.close()

# Airline-wise Sentiment Breakdown
airline_sent = df_clean.groupBy("airline", "airline_sentiment").count()
airline_sent_pd = airline_sent.toPandas()
pivot = airline_sent_pd.pivot(index='airline', columns='airline_sentiment', values='count').fillna(0)
pivot.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title('Sentiment by Airline')
plt.ylabel('Count')
plt.xlabel('Airline')
plt.savefig('dataset/airline_sentiment.png')
plt.close()

# Negative Reasons Analysis
neg_reason = df_clean.groupBy("negativereason").count().orderBy(col("count").desc())
neg_reason_pd = neg_reason.toPandas()
plt.figure(figsize=(10,5))
sns.barplot(x='negativereason', y='count', data=neg_reason_pd)
plt.title('Negative Reason Distribution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('dataset/neg_reason_distribution.png')
plt.close()

# Retweet count summary
df_clean.selectExpr("max(retweet_count) as max_retweets", "avg(retweet_count) as avg_retweets").show()

# Export preprocessed data
df_clean.toPandas().to_csv('dataset/twitter_sentiment_clean.csv', index=False)

print("All analyses and visualizations saved in 'dataset/' folder. Check PNG files and CSV for details.")
