import numpy as np
import pandas as pd
from perceptron import Perceptron


def classify(x: list, ppn: Perceptron) -> str:
    """
    Classify the given data
    :param list x: list of data to classify
    :param Perceptron ppn: Perceptron to use
    :return: str - classification
    """
    if ppn.predict(x) == -1:
        return "Iris-setosa"
    else:
        return "Iris-versicolor"


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

df.tail()

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 1, 2, 3]].values

ppn = Perceptron(eta=0.01, n_iter=50)
ppn.fit(X, y)

print(ppn.errors_)
print(ppn.w_)
print(classify([5.2, 2.7, 3.9, 1.4], ppn)) # Iris-versicolor
