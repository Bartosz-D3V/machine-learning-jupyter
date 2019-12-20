from pandas import DataFrame


def del_columns(X, columns):
    x_copy = DataFrame.copy(X)
    return x_copy.drop(columns, axis=1)
