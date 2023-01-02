from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Recommendation').getOrCreate()

movies = spark.read.csv('data/movies.csv', header=True)
ratings = spark.read.csv('data/ratings.csv', header=True)

ratings.show()