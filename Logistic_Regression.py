import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

ad_data = pd.read_csv('advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()
sns.histplot(ad_data['Age'], bins=30)
sns.jointplot(x='Age', y='Area Income',data=ad_data)
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde')
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage', data=ad_data, color='green')
sns.pairplot(ad_data, hue ='Clicked on Ad')
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LogisticRegression()
predicted = cross_validation.cross_val_predict(model, X_train, y_train, cv=3)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(classification_report(y_test,pred))
print(classification_report(y_test,predicted))
