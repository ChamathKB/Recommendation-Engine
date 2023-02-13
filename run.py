from pyspark.sql.functions import col, explode
from model import Model, Tune
import consumer
import producer

model_path = '/model/model'

best_model = Model.load_model(model_path)

# als = ALS(userCol='userId',
#                 itemCol='movieId',
#                 ratingCol='rating',
#                 nonnegative=True, implicitPrefs=False, coldStartStrategy='drop')

# ranks = [10, 50, 100, 150]
# regparams = [.01, .05, .1, .15]

# evaluator, param_grid  = Tune.tune(als, ranks, regparams, 'rmse', 'ratings', 'prediction')

# predict = Model.predict(model=model, evaluator=evaluator)



n_recommendations = best_model.recommendForAllUsers(10)
n_recommendations.limit(10).show()

n_recommendations = n_recommendations.withColumn('rec_exp', explode('recommendations')) \
                                     .select('userId', col('rec_exp.movieId'), col("rec_exp.rating"))

n_recommendations.limits(10).show()