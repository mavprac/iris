import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = datasets.load_iris()
x,y = iris.data,iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size =.3 ,random_state = 42)

sv = SVC()
sv.fit(x_train,y_train)
sv_pred = sv.predict(x_test)
sv_ac = accuracy_score(y_test,sv_pred)

print("accuracy is ",sv_ac)
