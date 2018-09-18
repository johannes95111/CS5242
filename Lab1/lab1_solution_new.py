import sys
import re
from math import log, sqrt
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row, Column
from pyspark.sql.functions import monotonically_increasing_id

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
#input_query = []
#with open('query.txt') as inputfile:
#    for line in inputfile:
#        input_query.append(line.strip().split('\n'))

stop_words = []
with open('stopwords.txt') as inputfile:
    for line in inputfile:
        stop_words.extend(line.strip().split('\n'))

df_query = spark.read.text("./query.txt").withColumn("query_id", monotonically_increasing_id()+1)
df_doc = spark.read.json("./kindle_review/xbg2.json")

#df = spark.read.json("./kindle_review/reviews_Kindle_Store.json")
n_doc = df_doc.count()
df_doc.printSchema()
df_doc.select("reviewText").show()
df_doc.select("summary").show()

###### step 1: compute TF of every word in a review
def udf_create_doc_rdd(row):
    doc_id = row.reviewerID + '_' + row.asin
    review_text = row.reviewText.lower()
    summary_text = row.summary.lower()
    info = re.findall(r'[\w]+', review_text) + re.findall(r'[\w]+', summary_text)
    clean_info = [w for w in info if w not in stop_words]
    return list((doc_id, w) for w in clean_info)
    
# doc_rdd RDD format: [(doc_id, word)...]
doc_rdd = df_doc.rdd.flatMap(udf_create_doc_rdd)

tf_pairs = doc_rdd.map(lambda x: (x, 1))
# tf_counts RDD format: [((doc_id, word), tf_count)...]
tf_counts = tf_pairs.reduceByKey(lambda n1, n2: n1 + n2)

###### Step 2: compute TF-IDF of every word w.r.t a document
df_pairs = tf_pairs.groupByKey().map(lambda x: (x[0][1], 1))
# df_counts RDD format: [(word, df_count)...]
df_counts = df_pairs.reduceByKey(lambda n1, n2: n1 + n2)

tf_df_temp = tf_counts.map(lambda x: (x[0][1], (x[0][0], x[1]))).join(df_counts)
# tf_df_counts RDD format: [((doc_id, word), (tf_count, df_count))...]
tf_df_counts = tf_df_temp.map(lambda x: ((x[1][0][0], x[0]), (x[1][0][1], x[1][1])))
# tf_idf RDD format: [((doc_id, word), tf_idf)...]
tf_idf = tf_df_counts.map(lambda x: ((x[0], ((1+log(x[1][0]))*log(n_doc/x[1][1])))))

###### Step 3: compute normalized TF-IDF of every word w.r.t a document
tf_idf_temp = tf_idf.map(lambda x: (x[0][0], x[1]**2))
# ss_tf_idf RDD format: [(doc_id, sum_of_squares_of_tf_idf)...]
ss_tf_idf = tf_idf_temp.reduceByKey(lambda n1, n2: (n1 + n2))
ss_tf_idf_sqrt = ss_tf_idf.map(lambda x: (x[0], sqrt(x[1])))
tf_idf_norm_temp = tf_idf.map(lambda x: (x[0][0], (x[0][1], x[1]))).join(ss_tf_idf_sqrt)
# tf_idf_norm RDD format: [((doc_id, word), tf_idf_norm)...]
tf_idf_norm = tf_idf_norm_temp.map(lambda x: ((x[0], x[1][0][0]), x[1][0][1]/x[1][1]))

###### Step 4: compute the relevance of each document w.r.t a query
def udf_create_query_rdd(row):
    query_id = 'query' + str(row.query_id)
    query_text = row.value.lower()
    info = re.findall(r'[\w]+', query_text)
    clean_info = [w for w in info if w not in stop_words]
    return list(((query_id, w) for w in clean_info))

# query_rdd RDD format: [(query_id, word)...]
query_rdd = df_query.rdd.flatMap(udf_create_query_rdd)
rev_query_rdd = query_rdd.map(lambda x: (x[1], x[0]))

temp1 = tf_idf_norm.map(lambda x: (0, (x[0][0],x[0][1],x[1])))
temp2 = query_rdd.groupByKey().map(lambda x: (0, x[0]))
# summary_rdd_zero RDD format: [((query_id, doc_id, word, tf_idf_norm),0)...]
summary_rdd_zero = temp1.join(temp2).map(lambda x:
    ((x[1][1],x[1][0][0],x[1][0][1],x[1][0][2]),(x[0])))
word_tf_idf_norm = tf_idf_norm.map(lambda x: (x[0][1], (x[0][0], x[1])))
# summary_rdd_one RDD format: [((query_id, doc_id, word, tf_idf_norm),1)...]
summary_rdd_one = rev_query_rdd.join(word_tf_idf_norm).map(lambda x:
    ((x[1][0],x[1][1][0],x[0],x[1][1][1]),1))
# summary_rdd RDD format: [((query_id, doc_id, word, tf_idf_norm),0/1)...]
summary_rdd = summary_rdd_zero.union(summary_rdd_one).reduceByKey(lambda n1, n2: (n1 + n2))
summary_rdd_temp = summary_rdd.map(lambda x: ((x[0][0], x[0][1]), (x[0][3], x[1])))
# summary_vec RDD format: [((query_id, doc_id), [(tf_idf_norm, 0/1)...])...]
summary_vec = summary_rdd_temp.groupByKey().map(lambda x: (x[0], list(x[1])))

def udf_cal_relevance(row):
    vec = row[1]
    sum_v1mv2, sum_v1mv1, sum_v2mv2 = 0, 0, 0
    for v in vec:
        sum_v1mv2 += v[0]*v[1]
        sum_v1mv1 += v[0]*v[0]
        sum_v2mv2 += v[1]*v[1]
    if sum_v1mv1 == 0 or sum_v2mv2 == 0:
        similarity = 0
    else:
        similarity = sum_v1mv2/(sqrt(sum_v1mv1)*sqrt(sum_v2mv2))
    return (row[0][0], (row[0][1], similarity))

# summary_rel RDD format: [(query_id, (doc_id, relevance))...]
summary_rel = summary_vec.map(udf_cal_relevance)

###### Step 5: sort and get top-k documents
top_k = 1
# summary_res RDD format: [(query_id, [(doc_id, relevance)...])...]
summary_res = summary_rel.groupByKey().map(lambda x: (x[0], list(x[1])))
summary_res_sorted = summary_res.map(lambda x: (x[0], sorted(x[1], key=lambda i: i[1], reverse=True)))
summary_res_top_k = summary_res_sorted.map(lambda x: (x[0], x[1][:top_k]))

# close spark session
spark.stop()
