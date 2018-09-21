import re
from math import log, sqrt
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import monotonically_increasing_id

###### step 0: configuration and setup
### define input files path
stop_words_filepath = './stopwords.txt'
query_filepath = './query.txt'
doc_filepath = './kindle_review/xaa.json'
output_filepath = './query_output.txt'
### define top-k number
top_k = 20
### initialize Spark Context and SQL Context
conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
### read in stop words as a list
stop_words = []
with open(stop_words_filepath) as inputfile:
    for line in inputfile:
        stop_words.extend(line.strip().split('\n'))
### read in query and document as SQL dataframe 
df_query = sqlContext.read.text(query_filepath).withColumn('query_id', monotonically_increasing_id()+1)
df_doc = sqlContext.read.json(doc_filepath)

###### step 1: compute TF of every word in a review
### user defined function to create document RDD
def udf_create_doc_rdd(row):
    doc_id = row.reviewerID + '_' + row.asin
    review_text = row.reviewText.lower()
    summary_text = row.summary.lower()
    info = re.findall(r'[\w]+', review_text) + re.findall(r'[\w]+', summary_text)
    clean_info = [w for w in info if w not in stop_words]
    return list((doc_id, w) for w in clean_info)
### doc_rdd RDD format: [(doc_id, word)...]
doc_rdd = df_doc.rdd.flatMap(udf_create_doc_rdd)
### tf_pairs RDD format: [((doc_id, word), 1)...]
tf_pairs = doc_rdd.map(lambda x: (x, 1))
### tf_counts RDD format: [((doc_id, word), tf_count)...]
tf_counts = tf_pairs.reduceByKey(lambda n1, n2: n1 + n2)

###### Step 2: compute TF-IDF of every word w.r.t a document
### get the number of reviews
n_doc = df_doc.count()
### df_pairs RDD format: [(word, 1)...]
df_pairs = tf_pairs.groupByKey().map(lambda x: (x[0][1], 1))
### df_counts RDD format: [(word, df_count)...]
df_counts = df_pairs.reduceByKey(lambda n1, n2: n1 + n2)
### tf_df_temp RDD format: []
tf_df_temp = tf_counts.map(lambda x: (x[0][1], (x[0][0], x[1]))).join(df_counts)
### tf_df_counts RDD format: [((doc_id, word), (tf_count, df_count))...]
tf_df_counts = tf_df_temp.map(lambda x: ((x[1][0][0], x[0]), (x[1][0][1], x[1][1])))
### tf_idf RDD format: [((doc_id, word), tf_idf)...]
tf_idf = tf_df_counts.map(lambda x: ((x[0], ((1+log(x[1][0]))*log(n_doc/x[1][1])))))

###### Step 3: compute normalized TF-IDF of every word w.r.t a document
### tf_idf_temp RDD format: [(doc_id, squres_of_tf_idf)...]
tf_idf_temp = tf_idf.map(lambda x: (x[0][0], x[1]**2))
### ss_tf_idf RDD format: [(doc_id, sum_of_squares_of_tf_idf)...]
ss_tf_idf = tf_idf_temp.reduceByKey(lambda n1, n2: (n1 + n2))
### ss_tf_idf_sqrt RDD format: [(doc_id, square_of_sum_of_squares_of_tf_idf)...]
ss_tf_idf_sqrt = ss_tf_idf.map(lambda x: (x[0], sqrt(x[1])))
### tf_idf_norm_temp RDD format: [(doc_id, ((word, tf_idf), square_of_sum_of_squares_of_tf_idf))]
tf_idf_norm_temp = tf_idf.map(lambda x: (x[0][0], (x[0][1], x[1]))).join(ss_tf_idf_sqrt)
### tf_idf_norm RDD format: [((doc_id, word), tf_idf_norm)...]
tf_idf_norm = tf_idf_norm_temp.map(lambda x: ((x[0], x[1][0][0]), x[1][0][1]/x[1][1]))

###### Step 4: compute the relevance of each document w.r.t a query
### user defined function to create query RDD
def udf_create_query_rdd(row):
    query_id = 'query' + str(row.query_id)
    query_text = row.value.lower()
    info = re.findall(r'[\w]+', query_text)
    clean_info = [w for w in info if w not in stop_words]
    return list(((query_id, w) for w in clean_info))
### query_rdd RDD format: [(query_id, word)...]
query_rdd = df_query.rdd.flatMap(udf_create_query_rdd)
### rev_query_rdd RDD format: [(word, query_id)...]
rev_query_rdd = query_rdd.map(lambda x: (x[1], x[0]))
### zero_tf_idf_norm RDD format: [(0, (doc_id, word, tf_idf_norm))...]
zero_tf_idf_norm = tf_idf_norm.map(lambda x: (0, (x[0][0],x[0][1],x[1])))
### zero_query_rdd RDD format: [(0, query_id)...]
zero_query_rdd = query_rdd.groupByKey().map(lambda x: (0, x[0]))
### summary_rdd_zero_temp RDD format: [(0/1, (doc_id, word, tf_idf_norm))...]
summary_rdd_zero_temp = zero_tf_idf_norm.join(zero_query_rdd)
# summary_rdd_zero RDD format: [((query_id, doc_id, word, tf_idf_norm), 0)...]
summary_rdd_zero = summary_rdd_zero_temp.map(lambda x: ((x[1][1],x[1][0][0],x[1][0][1],x[1][0][2]),(x[0])))
### word_tf_idf_norm RDD format: [(word, (doc_id, tf_idf_norm))]
word_tf_idf_norm = tf_idf_norm.map(lambda x: (x[0][1], (x[0][0], x[1])))
### summary_rdd_one_temp RDD format: [(word, (query_id, (doc_id, tf_idf_norm)))...]
summary_rdd_one_temp = rev_query_rdd.join(word_tf_idf_norm)
### summary_rdd_one RDD format: [((query_id, doc_id, word, tf_idf_norm), 1)...]
summary_rdd_one = summary_rdd_one_temp.map(lambda x: ((x[1][0],x[1][1][0],x[0],x[1][1][1]),1))
### summary_rdd_union RDD format: [((query_id, doc_id, word, tf_idf_norm), 0/1)...]
summary_rdd_union = summary_rdd_zero.union(summary_rdd_one)
### summary_rdd RDD function: [((query_id, doc_id, word, tf_idf_norm), 0/1)...]
summary_rdd = summary_rdd_union.reduceByKey(lambda n1, n2: (n1 + n2))
### summary_rdd_temp RDD format: [((query_id, doc_id), (tf_idf_norm, 0/1))]
summary_rdd_temp = summary_rdd.map(lambda x: ((x[0][0], x[0][1]), (x[0][3], x[1])))
### summary_vec RDD format: [((query_id, doc_id), [(tf_idf_norm, 0/1)...])...]
summary_vec = summary_rdd_temp.groupByKey().map(lambda x: (x[0], list(x[1])))
### user defined function to calculate relevance of two vectors
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
### summary_rel RDD format: [(query_id, (doc_id, relevance))...]
summary_rel = summary_vec.map(udf_cal_relevance)

###### Step 5: sort and get top-k documents
### summary_res RDD format: [(query_id, [(doc_id, relevance)...])...]
summary_res = summary_rel.groupByKey().map(lambda x: (x[0], list(x[1])))
### sort summary_res RDD as summary_res_sorted RDD
summary_res_sorted = summary_res.map(lambda x: (x[0], sorted(x[1], key=lambda i: i[1], reverse=True)))
### get top-k documents as summary_res_top_k
summary_res_top_k = summary_res_sorted.map(lambda x: (x[0], x[1][:top_k]))
### collect results from summary_res_top_k and as a list
collect_res_top_k = summary_res_top_k.collect()
### sort results by query_id
collect_res_top_k.sort(key=lambda i: i[0])
### write results to output text file
with open(output_filepath, 'w') as outputfile:
    outputfile.write("query_id, document_id, relevance_score\n")
    for res in collect_res_top_k:
        query_id = res[0]
        for rec in res[1]:
            line = '{}, {}, {}\n'.format(query_id, rec[0], rec[1])
            outputfile.write(line)

### close Spark Context
sc.stop()
