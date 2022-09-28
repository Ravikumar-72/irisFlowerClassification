import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('iris.csv')

X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y = df['Species']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=500)

from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
Y_train=labelencoder_y.fit_transform(Y_train)
Y_test=labelencoder_y.fit_transform(Y_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors =5, metric="minkowski",p=2)
pred = classifier.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score
print('Accuracy Score: ',accuracy_score(Y_test,pred))

pickle.dump(classifier,open('model-iris1.pkl','wb'))