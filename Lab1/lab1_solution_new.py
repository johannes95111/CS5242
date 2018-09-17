import sys
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row, Column

## initialize spark context
#conf = SparkConf()
#sc = SparkContext(conf=conf)

# initialize spark session
spark = SparkSession \
    .builder \
    .appName("Python Spark Program") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# read input data
input_query = []
with open('query.txt') as inputfile:
    for line in inputfile:
        input_query.append(line.strip().split('\n'))

stop_words = []
with open('stopwords.txt') as inputfile:
    for line in inputfile:
        stop_words.extend(line.strip().split('\n'))

df = spark.read.json("./kindle_review/xbg2.json")
#df = spark.read.json("./kindle_review/reviews_Kindle_Store.json")
df.printSchema()
df.select("reviewText").show()
df.select("summary").show()

###### step 1: compute TF of every word in a review
def udf_create_doc(row):
    doc_id = row.reviewerID + '_' + row.asin
    review_text = row.reviewText.lower()
    summary_text = row.summary.lower()
    info = re.findall(r'[\w]+', review_text) + re.findall(r'[\w]+', summary_text)
    clean_info = [w for w in info if w not in stop_words]
    return list((doc_id, w) for w in clean_info)
    

doc_rdd = df.rdd.flatMap(udf_create_doc)
#doc_df = doc_rdd.toDF(["doc_id", "info"])
tf_pairs = doc_rdd.map(lambda x: (x, 1))
tf_counts = tf_pairs.reduceByKey(lambda n1, n2: n1 + n2)

###### Step 2: compute TF-IDF of every word w.r.t a document

df_pairs = tf_pairs.groupByKey().map(lambda x: (x[0][1], 1))
df_counts = df_pairs.reduceByKey(lambda n1, n2: n1 + n2)

# close spark session
spark.stop()
