import pytest
from pyspark.sql import SparkSession
from data_pipeline import sparsity

spark = SparkSession.builder.appName('Recommendation').getOrCreate()

data = spark.read.csv('data/ratings.csv', header=True)

def test_sparsity(data):
    assert 'dataframe' in sparsity(data=data)