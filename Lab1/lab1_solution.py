import sys
import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

# initialize spark context
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

df = spark.read.json("./kindle_review/xbg.json")
#df = spark.read.json("./kindle_review/reviews_Kindle_Store.json")
df.select("reviewText").show()

# step 1: compute TF of every word in a review
def split_review(review):
    
    
test = df.select("reviewText").rdd.map(lambda l: l[0])
print(test.take(1))


## close spark context
#sc.stop()
#
## close spark session
spark.stop()
