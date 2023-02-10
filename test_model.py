import pytest
from pyspark.sql import SparkSession
from model import Model, Tune, Utils

spark = SparkSession.builder.appName('Recommendation').getOrCreate()

data = spark.read.csv('data/ratings.csv', header=True)

def test_sparsity(data):
    sparsity = Utils.sparsity()
    assert type(sparsity(data)) == int