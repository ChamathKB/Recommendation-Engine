from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import col, explode
import pandas as pd

spark = SparkSession.builder.appName('Recommendation').getOrCreate()
sc = SparkContext

movies = spark.read.csv('data/movies.csv', header=True)
ratings = spark.read.csv('data/ratings.csv', header=True)

# ratings.show()

ratings.printSchema()

ratings = ratings.withColumn('userId', col('userId').cast('integer')).\
                  withColumn('movieId', col('movieId').cast('integer')).\
                  withColumn('rating', col('rating').cast('float')).\
                  drop('timestamp')

# ratings.show()

numerator = ratings.select('rating').count()

num_users = ratings.select('userId').distinct().count()
num_movies = ratings.select('movieId').distinct().count()

denominator = num_movies * num_users

sparsity = (1.0 - (numerator * 1.0)/denominator) * 100

print("The rating dataframe is ", "%.2f" % sparsity + "% empty.")


userId_ratings = ratings.groupBy('userId').count().orderBy('count', ascending=False)
# userId_ratings.show()

movieId_ratings = ratings.groupBy('movieId').count().orderBy('count', ascending=False)
# movieId_ratings.show()

