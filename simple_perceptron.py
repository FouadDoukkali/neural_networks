import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
 
iris = load_iris()

df_iris = pd.DataFrame(iris.data, columns = iris.feature_names)

X = df_iris.iloc[:,2:]
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

pcpt = Perceptron(random_state = 4) 
pcpt.fit(X_train,y_train)
ypred = pcpt.predict(X_test)

accuracy = accuracy_score( y_test, ypred) 
print(accuracy)
# Accuracy is 1, that is to be expected dataset is small

cm = confusion_matrix(y_test, ypred)
# Confusion matrix displays no mislabelling, which is worrisome.
print(cm)
