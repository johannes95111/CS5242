import sys
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row, Column

# initialize spark context
conf = SparkConf()
sc = SparkContext(conf=conf)

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
    info = row.reviewText.lower() + ' ' + row.summary.lower()
    info = ' '.join([w for w in info.split() if w not in stop_words])
    return Row(doc_id, info)

doc_rdd = df.rdd.map(udf_create_doc)
doc_df = doc_rdd.toDF(["doc_id", "info"])

words = doc_df.rdd.map(lambda d: (d["doc_id"], re.findall(r'[\w]+', d["info"])))
pairs = [sc.parallelize(list(map(lambda x: (x, 1), w[1]))) for w in words.collect()]
counts = [p.reduceByKey(lambda n1, n2: n1 + n2) for p in pairs]

def udf_create_doc2(row):
    doc_id = row.reviewerID + '_' + row.asin
    review_text = row.reviewText.lower()
    summary_text = row.summary.lower()
    info = re.findall(r'[\w]+', review_text) + re.findall(r'[\w]+', summary_text)
    clean_info = [w for w in info if w not in stop_words]
    return Row(doc_id, clean_info)
    

doc_rdd2 = df.rdd.map(udf_create_doc2)
doc_df2 = doc_rdd2.toDF(["doc_id", "info"])

#words2 = sc.parallelize(doc_df2.select("info").collect())
#pairs2 = words2.map(lambda d: Row(pair=[(w, 1) for w in d[0]]))
words2 = doc_df2.select("info").collect()
pairs2 = [sc.parallelize(list(map(lambda x: (x, 1), w[0]))) for w in words2]
counts2 = [p.reduceByKey(lambda n1, n2: n1 + n2) for p in pairs2]

#def udf_count_tf2(row):
#    pairs = [(w, 1) for w in row.info]
#    #counts = [p.reduceByKey(lambda n1, n2:n1 + n2) for p in pairs]
#    return (row.doc_id, pairs)
#
#test_res = doc_df2.rdd.map(udf_count_tf2)
#test_df = test_res.toDF(["doc_id", "pairs"])
#
#word_pairs = doc_df2.select("info").rdd.map(lambda d: [(w, 1) for w in d[0]])
##word_counts = p.reduceByKey(lambda d: [dn1 + n2))
#
#
##def udf_count_tf(row):
##    info = re.findall(r'[\w]+', row.info)
##    pairs = list(map(lambda w: (w, 1), info))
##    return (row.doc_id, info, pairs)
#
#def udf_count_tf(row):
#    info = re.findall(r'[\w]+', row.info)
#    pairs = list(map(lambda w: (w, 1), info))
#    p = sc.parallelize(pairs)
#    counts = p.reduceByKey(lambda n1, n2: n1 + n2)
#    print(counts)
#    return counts
#
## extract words from documents
#doc_df2 = doc_df.rdd.map(udf_count_tf)    
#
#words = doc_df.select("info").rdd.map(lambda d: re.findall(r'[\w]+', d[0]))
#words2 = doc_df.rdd.map(lambda d: (d["doc_id"], re.findall(r'[\w]+', d["info"])))
#doc_df2 = doc_df.withColumn("words", words)
## count the number of words
#pairs2 = [sc.parallelize(list(map(lambda x: (x, 1), w[1]))) for w in words2.collect()]
#counts = [p.reduceByKey(lambda n1, n2: n1 + n2) for p in pairs2]
#pairs = words2.flatMapValues(lambda w: (w, 1))

###### Step 2: compute TF-IDF of every word w.r.t a document
doc_df_clt = doc_df2.collect()
#pairs_wrt_doc = list(list(map(lambda x: (x, 1), w["info"])) for w in doc_df_clt)
pairs_wrt_doc = sc.parallelize(list(map(lambda x: (x, w["doc_id"]), w["info"])) for w in doc_df_clt)
pairs_wrt_doc_red = pairs_wrt_doc.reduce(lambda a, b: a + b)
pairs_wrt_doc_rdd = sc.parallelize(pairs_wrt_doc_red)
pairs_wrt_doc_counts = pairs_wrt_doc_rdd.groupByKey().map(lambda x: (x[0], len(list(x[1]))))

# close spark context
sc.stop()

# close spark session
spark.stop()
