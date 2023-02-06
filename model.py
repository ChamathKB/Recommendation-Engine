from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import col, explode

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


class Model():

    def __init__(self, data):
        self.data = data

    def model(self):
        """create pyspark model

        Returns:
            Any: train, test data and als model
        """        
        train, test = self.data.randomSplit([0.8, 0.2], seed=1)

        als = ALS(userCol='userId',
                itemCol='movieId',
                ratingCol='rating',
                nonnegative=True, implicitPrefs=False, coldStartStrategy='drop')

        return train, test, als
        
    def predict(model, data, evaluator):
        """predict from trained model

        Args:
            model (object): model use for prediction
            data (dataframe): test data
            evaluator (object): evaluator

        Returns:
            string: RMSE
            list: test prediciton
        """        
        test_predictions = model.transform(data)

        RMSE = evaluator.evaluate(test_predictions)

        print(RMSE)

        return test_predictions, RMSE

class Tune():
    def __init__(self, model):
        self.model = model

    def tune(self, ranks, regparams, metric, labelcol, predictioncol):
        """finetune model

        Args:
            ranks (array): rank range
            regparams (array): regparam range
            metric (string): accuracy metric
            labelcol (string): label column
            predictioncol (string): target column

        Returns:
            Any: evaluate model
            Array: param_grid
        """        
        param_grid = ParamGridBuilder() \
                    .addGrid(self.model.rank, ranks) \
                    .addGrid(ranks.regParam, regparams) \
                    .build()

        evaluator = RegressionEvaluator(metricName=metric, labelCol=labelcol, predictionCol=predictioncol)

        print("Number of models to test: ", len(param_grid))

        return evaluator, param_grid

        
    def best_model(self):
        """select best model

        Returns:
            object: best model
        """        
        best_model = self.model.bestModel

        print(type(best_model))

        print("Best model ->")

        print("  Rank: ", best_model._java_obj.parent().getRank())

        print("  MaxIter: ", best_model._java_obj.parent().getMaxIter())

        print("  RegParam: ", best_model._java_obj.parent().getRegParam())

        return best_model

class Utils():

    def sparsity(data):
        """calculate sparsity

        Args:
            data (_type_): dataset for train/test model

        Returns:
            float: sparsity value
        """            
        numerator = data.select('rating').count()

        num_users = data.select('userId').distinct().count()
        num_movies = data.select('movieId').distinct().count()

        denominator = num_movies * num_users

        sparsity = (1.0 - (numerator * 1.0)/denominator) * 100

        print("The rating dataframe is ", "%.2f" % sparsity + "% empty.")

        return sparsity