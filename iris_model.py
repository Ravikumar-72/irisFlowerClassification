import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('iris.csv')
x = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2,random_state=50)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from  sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
pred = classifier.fit(X_train,Y_train)

def acc(Y_test,pred):
    from sklearn.metrics import accuracy_score
    acc_s = accuracy_score(Y_test,pred)
    return acc_s

Acc = acc(Y_test,pred)

pickle.dump(classifier,open('model-iris.pkl','wb'))