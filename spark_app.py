import os
import sys
import textwrap

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Spark session (portable) ----------------
spark = (
    SparkSession.builder
    .appName("Twitter Sentiment Analysis")
    .master("local[*]")   # run locally in Codespaces/Actions
    .getOrCreate()
)

# ---------------- Data load ----------------
SRC = "dataset/twitter_sentiment.csv"
OUT_DIR = "dataset"
os.makedirs(OUT_DIR, exist_ok=True)

df = spark.read.csv(SRC, header=True, inferSchema=True)

# ---------------- Cleaning ----------------
# 1) drop URLs, mentions, hashtags; 2) lowercase; 3) keep letters and spaces; 4) collapse spaces
df_clean = df.withColumn(
    "clean_text",
    lower(
        regexp_replace(
            regexp_replace(
                regexp_replace(col("text"), r"http\\S+", ""),
                r"@[A-Za-z0-9_]+", ""
            ),
            r"#[A-Za-z0-9_]+", ""
        )
    )
)
df_clean = df_clean.withColumn("clean_text", regexp_replace(col("clean_text"), r"[^a-zA-Z\\s]", ""))
df_clean = df_clean.withColumn("clean_text", regexp_replace(col("clean_text"), r"\\s+", " "))

# ---------------- Plot style helpers ----------------
sns.set_theme(style="whitegrid", context="talk")

def _save_fig(path, fig=None, dpi=180):
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _wrap(labels, width=18):
    return [textwrap.fill(str(x), width=width) for x in labels]

# ---------------- 1) Sentiment distribution ----------------
# Fix order and annotate bars; prevent overlapping by constrained layout
sentiment_counts = (
    df_clean.groupBy("airline_sentiment").count()
)

sentiment_pd = sentiment_counts.toPandas()
# Standardize sentiment order if present
order = [s for s in ["positive", "negative", "neutral"] if s in sentiment_pd["airline_sentiment"].unique()]
if not order:
    order = sentiment_pd.sort_values("count", ascending=False)["airline_sentiment"].tolist()

fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
sns.barplot(x="airline_sentiment", y="count", data=sentiment_pd, order=order, palette="pastel", ax=ax)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of tweets")
ax.set_title("Overall Sentiment Distribution", pad=10)

# Annotate bars
for c in ax.containers:
    ax.bar_label(c, fmt="%d", padding=3)

_save_fig(os.path.join(OUT_DIR, "sentiment_distribution.png"), fig)

# ---------------- 2) Sentiment by airline (stacked) ----------------
# Limit to top N airlines to avoid overcrowding; wrap and rotate x ticks; external legend
TOP_AIRLINES = 8
airline_sent = df_clean.groupBy("airline", "airline_sentiment").count()
airline_sent_pd = airline_sent.toPandas()

if not airline_sent_pd.empty:
    # Select top airlines by total volume
    tot = airline_sent_pd.groupby("airline", as_index=False)["count"].sum().sort_values("count", ascending=False)
    top_air = tot.head(TOP_AIRLINES)["airline"].tolist()
    subset = airline_sent_pd[airline_sent_pd["airline"].isin(top_air)].copy()

    # Pivot to stacked bars with a consistent sentiment column order where possible
    cols = [c for c in ["positive", "negative", "neutral"] if c in subset["airline_sentiment"].unique()]
    pivot = (
        subset.pivot(index="airline", columns="airline_sentiment", values="count")
        .fillna(0)
        .reindex(columns=cols)
        .loc[top_air]
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    pivot.plot(kind="bar", stacked=True, ax=ax, color=sns.color_palette("husl", len(pivot.columns)))
    ax.set_xlabel("Airline")
    ax.set_ylabel("Tweet count")
    ax.set_title("Sentiment by Airline (Top {})".format(TOP_AIRLINES), pad=10)

    # Wrap and tilt airline names
    ax.set_xticklabels(_wrap(pivot.index, 14), rotation=15, ha="right")

    # Put legend outside to avoid overlap
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    _save_fig(os.path.join(OUT_DIR, "airline_sentiment.png"), fig)

# ---------------- 3) Negative reason analysis ----------------
# Use horizontal bars with wrapped labels to avoid x‑axis collisions
TOP_REASONS = 12
neg_reason = df_clean.groupBy("negativereason").count()
neg_reason_pd = (
    neg_reason.toPandas()
    .dropna(subset=["negativereason"])
    .sort_values("count", ascending=False)
    .head(TOP_REASONS)
)

if not neg_reason_pd.empty:
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    # Wrap reason labels for y ticks
    y_labels = _wrap(neg_reason_pd["negativereason"].tolist(), width=28)
    sns.barplot(x="count", y=y_labels, data=neg_reason_pd.assign(negativereason_wrapped=y_labels),
                orient="h", palette=sns.color_palette("crest", n_colors=len(neg_reason_pd)), ax=ax)

    ax.set_xlabel("Number of tweets")
    ax.set_ylabel("Negative reason")
    ax.set_title("Top Negative Reasons", pad=10)

    # Annotate on the bars
    for c in ax.containers:
        ax.bar_label(c, fmt="%d", padding=3)

    _save_fig(os.path.join(OUT_DIR, "neg_reason_distribution.png"), fig)

# ---------------- 4) Retweet stats ----------------
# Show summary in console
df_clean.selectExpr("max(retweet_count) as max_retweets", "avg(retweet_count) as avg_retweets").show()

# ---------------- 5) Export cleaned data ----------------
# If the dataset is very large, consider writing via Spark to a single CSV with coalesce(1)
# Here we assume modest size and export via pandas for a single-file output
clean_pd_cols = [c for c in df_clean.columns if c in {"tweet_id", "airline", "airline_sentiment", "negativereason", "clean_text", "retweet_count"}]
clean_pd = df_clean.select(*clean_pd_cols).toPandas()
clean_pd.to_csv(os.path.join(OUT_DIR, "twitter_sentiment_clean.csv"), index=False)

print("All analyses and visualizations saved in 'dataset/' folder. Check PNG files and CSV for details.")
