


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