from pyspark.sql.functions import col, explode
from model import Model, Tune
import consumer
import producer


best_model = Model.load_model('model/ALS_model/bestModel')


n_recommendations = best_model.recommendForAllUsers(10)
n_recommendations.limit(10).show()

n_recommendations = n_recommendations.withColumn('rec_exp', explode('recommendations')) \
                                     .select('userId', col('rec_exp.movieId'), col("rec_exp.rating"))

n_recommendations.limits(10).show()