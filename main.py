from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import col, explode

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd

from model import Model, Tune, Utils

spark = SparkSession.builder.appName('Recommendation').getOrCreate()
sc = SparkContext

movies = spark.read.csv('data/movies.csv', header=True)
ratings = spark.read.csv('data/ratings.csv', header=True)

# ratings.show()

#ratings.printSchema()

ratings = ratings.withColumn('userId', col('userId').cast('integer')).\
                  withColumn('movieId', col('movieId').cast('integer')).\
                  withColumn('rating', col('rating').cast('float')).\
                  drop('timestamp')

# ratings.show()


sparsity = Utils.sparsity(ratings)

# type(als)

train, test, als = Model.model(ratings)


ranks = [10, 50, 100, 150]
regparams = [.01, .05, .1, .15]

evaluator, param_grid  = Tune.tune(als, ranks, regparams, 'rmse', 'ratings', 'prediction')

cv = CrossValidator(estimator=als,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=5)

print(cv)

model = cv.fit(train)


best_model = Model.best_model(model)

prediction = Model.predict(best_model, test, evaluator)


n_recommendations = best_model.recommendForAllUsers(10)
n_recommendations.limit(10).show()

n_recommendations = n_recommendations.withColumn('rec_exp', explode('recommendations')) \
                                     .select('userId', col('rec_exp.movieId'), col("rec_exp.rating"))

n_recommendations.limits(10).show()

n_recommendations.join(movies, on='movieId').filter('userId = 100').show()

ratings.join(movies, on='movieId').filter('userId = 100').sort('rating', ascending=False).limit(10).show()