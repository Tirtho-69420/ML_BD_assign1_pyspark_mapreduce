import sys
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, split, element_at, avg, regexp_replace, lower, desc, abs, length
from pyspark.sql.types import StringType, FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Normalizer


spark = SparkSession.builder \
    .appName("Assignment1_Spark") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Loading dataset from books_dataset/ ...")
books_rdd = spark.sparkContext.wholeTextFiles("books_dataset/*.txt")
books_df = books_rdd.toDF(["file_path", "text"])


books_df = books_df.withColumn("file_name", element_at(split(col("file_path"), "/"), -1))


print("\n--- Q10: Metadata Extraction ---")

def extract_metadata(text):
    title_m = re.search(r"Title:\s+(.*)", text)
    date_m = re.search(r"Release Date:.*?(\d{4})", text, re.DOTALL)
    lang_m = re.search(r"Language:\s+(.*)", text)
    enc_m = re.search(r"Character set encoding:\s+(.*)", text)
    
    return (
        title_m.group(1).strip() if title_m else "Unknown",
        date_m.group(1) if date_m else None,
        lang_m.group(1).strip() if lang_m else "Unknown",
        enc_m.group(1).strip() if enc_m else "Unknown"
    )

meta_schema = "title STRING, release_year STRING, language STRING, encoding STRING"
meta_udf = udf(extract_metadata, meta_schema)

df_meta = books_df.withColumn("meta", meta_udf("text")) \
                  .select("file_name", "meta.*", "text")

print("\n[Output] Books released per year:")
df_meta.filter(col("release_year").isNotNull()) \
       .groupBy("release_year").count() \
       .orderBy("release_year").show(10)


print("\n[Output] Most common language:")
df_meta.groupBy("language").count().orderBy(desc("count")).show(1)


print("\n[Output] Average Title Length:")
df_meta.withColumn("title_len", length(col("title"))).agg(avg("title_len")).show()


print("\n--- Q11: TF-IDF & Similarity ---")

clean_df = df_meta.withColumn("clean_text", lower(col("text")))
clean_df = clean_df.withColumn("clean_text", regexp_replace(col("clean_text"), "[^a-zA-Z\\s]", ""))

tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
words_df = tokenizer.transform(clean_df)

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filtered_df = remover.transform(words_df)

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=2000)
featurized = hashingTF.transform(filtered_df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
model = idf.fit(featurized)
rescaled = model.transform(featurized)

normalizer = Normalizer(inputCol="features", outputCol="normFeatures")
norm_data = normalizer.transform(rescaled)


target_file = "200.txt"

available_files = [row.file_name for row in norm_data.select("file_name").limit(1).collect()]
if norm_data.filter(col("file_name") == target_file).count() == 0 and available_files:
    target_file = available_files[0]
    print(f"Note: 200.txt not found. Using {target_file} for similarity example.")

target_book = norm_data.filter(col("file_name") == target_file).first()

if target_book:
    target_vec = target_book["normFeatures"]
    def cosine_sim(v):
        return float(target_vec.dot(v))
    
    sim_udf = udf(cosine_sim, FloatType())
    
    print(f"\n[Output] Top 5 books similar to {target_file}:")
    norm_data.withColumn("similarity", sim_udf("normFeatures")) \
             .filter(col("file_name") != target_file) \
             .orderBy(desc("similarity")) \
             .select("file_name", "similarity") \
             .show(5, truncate=False)


print("\n--- Q12: Author Influence Network ---")

def get_author(text):
    m = re.search(r"Author:\s+(.*)", text)
    return m.group(1).strip() if m else None

auth_df = df_meta.withColumn("author", udf(get_author, StringType())("text")) \
                 .filter(col("author").isNotNull() & col("release_year").isNotNull()) \
                 .select("author", col("release_year").cast("int"))

df1 = auth_df.withColumnRenamed("author", "a1").withColumnRenamed("release_year", "y1")
df2 = auth_df.withColumnRenamed("author", "a2").withColumnRenamed("release_year", "y2")

edges = df1.crossJoin(df2).filter((col("a1") != col("a2")) & (abs(col("y1") - col("y2")) <= 5))

print("\n[Output] Top 5 Authors by Out-Degree (Influencers):")
edges.groupBy("a1").count().orderBy(desc("count")).show(5)

print("\n[Output] Top 5 Authors by In-Degree (Influenced):")
edges.groupBy("a2").count().orderBy(desc("count")).show(5)

spark.stop()
