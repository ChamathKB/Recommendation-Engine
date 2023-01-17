from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import col, explode

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd

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

def sparsity(data):        
    numerator = data.select('rating').count()

    num_users = data.select('userId').distinct().count()
    num_movies = data.select('movieId').distinct().count()

    denominator = num_movies * num_users

    sparsity = (1.0 - (numerator * 1.0)/denominator) * 100

    return print("The rating dataframe is ", "%.2f" % sparsity + "% empty.")

sparsity(ratings)


# userId_ratings = ratings.groupBy('userId').count().orderBy('count', ascending=False)
# userId_ratings.show()

# movieId_ratings = ratings.groupBy('movieId').count().orderBy('count', ascending=False)
# movieId_ratings.show()

def model(data):
    train, test = data.randomSplit([0.8, 0.2], seed=1)

    als = ALS(userCol='userId',
            itemCol='movieId',
            ratingCol='rating',
            nonnegative=True, implicitPrefs=False, coldStartStrategy='drop')

    return train, test, als

# type(als)

train, test, als = model(ratings)

def tune(model, ranks, regparams, metric, labelcol, predictioncol):
    param_grid = ParamGridBuilder() \
                .addGrid(model.rank, ranks) \
                .addGrid(ranks.regParam, regparams) \
                .build()

    evaluator = RegressionEvaluator(metricName=metric, labelCol=labelcol, predictionCol=predictioncol)

    print("Number of models to test: ", len(param_grid))

    return evaluator, param_grid

ranks = [10, 50, 100, 150]
regparams = [.01, .05, .1, .15]

evaluator, param_grid  = tune(als, ranks, regparams, 'rmse', 'ratings', 'prediction')

cv = CrossValidator(estimator=als,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=5)

print(cv)

model = cv.fit(train)

def best_model(model):
    best_model = model.bestModel

    print(type(best_model))

    print("Best model ->")

    print("  Rank: ", best_model._java_obj.parent().getRank())

    print("  MaxIter: ", best_model._java_obj.parent().getMaxIter())

    print("  RegParam: ", best_model._java_obj.parent().getRegParam())

    return best_model



best_model = best_model(model)

def predict(model, data, evaluator):
    test_predictions = model.transform(data)

    RMSE = evaluator.evaluate(test_predictions)

    print(RMSE)

    # test_predictions.show()

    return test_predictions, RMSE

predict(best_model, test, evaluator)


n_recommendations = best_model.recommendForAllUsers(10)
n_recommendations.limit(10).show()

n_recommendations = n_recommendations.withColumn('rec_exp', explode('recommendations')) \
                                     .select('userId', col('rec_exp.movieId'), col("rec_exp.rating"))

n_recommendations.limits(10).show()

n_recommendations.join(movies, on='movieId').filter('userId = 100').show()

ratings.join(movies, on='movieId').filter('userId = 100').sort('rating', ascending=False).limit(10).show()