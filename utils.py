def dot_product(x: list, y: list) -> int:
    """
    Calculate the dot product of two vectors
    :param list x: Vector 1
    :param list y: Vector 2
    :return: list
    """
    result = 0
    for i in range(len(x)):
        result += x[i] * y[i]
    return result

