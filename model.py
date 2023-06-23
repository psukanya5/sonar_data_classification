import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#loadding dataset to pandas dataframe
sonar_data = pd.read_csv('E:\data science projects\Project_1_Rock_Mine_Prediction\sonar data.csv',header = None)
#separating data and labels(Rock nd Mine)
X = sonar_data.drop(columns = 60, axis= 1)
y = sonar_data[60]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 2)
# model training (logistic regression)
model_1 = LogisticRegression()
#testing Logistic Regression Model with training data
model_1.fit(X_train, y_train)
with open('model.pkl','wb') as f:
    pickle.dump(model_1,f)


