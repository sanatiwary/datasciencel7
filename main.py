import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
print(dataset.info())

#features (input) - country, age, salary
#target (output) - purchased?

X = dataset.iloc[:, :-1].values # all except last
y = dataset.iloc[:, -1].values # pick last column

print("features: \n", X)
print("targets: \n", y)
print()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

#applying imputer to age and salary -- replacing nan with mean values
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

print("after imputing: \n", X)
print()

