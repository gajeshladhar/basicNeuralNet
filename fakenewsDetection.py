#Project
#Fake News Detection......

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

stopwords=list(stopwords.words('english'))

dataset=pd.read_csv("news.csv")
dataset=dataset.iloc[:,1:]

data=dataset.values[:,1:2]
labels=dataset.values[:,2:3]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.7)


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
temp=[]
for x,y in enumerate(data):
    for p in (tokenizer.tokenize(str(y[0]))):
        temp.append(p)
        

corpus=set(temp)
temp=[]
for some in corpus:
    if some not in stopwords:
        temp.append(some)
corpus=temp   
mp=dict(enumerate(corpus))
mp=dict([(v,k) for k,v in mp.items()])
    


train_vector=np.zeros(shape=(x_train.shape[0],len(corpus)))

for j,i in enumerate(x_train):
    for some in tokenizer.tokenize(i[0]) :
        if some not in stopwords:
            train_vector[j][mp[some]]=1


y_train=y_train=="REAL"
y_trainx=np.zeros(shape=(y_train.shape[0]))
for i,d in enumerate(y_train):
    if d :
        y_trainx[i]=1
    
y_train=y_trainx
y_train[0]

# Decision Tree Classifier..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

model=GradientBoostingClassifier(learning_rate=0.09,warm_start=True,n)
acc=[]
for i in range(1,10):
    model.n_estimators=i
    model.fit(train_vector[:900],y_train[:900])
    acc.append(accuracy_score(model.predict(train_vector[1500:]),y_train[1500:]))
    
#model=DecisionTreeClassifier()
model.fit(train_vector[1000:1900],y_train[1000:1900])

from sklearn.metrics import confusion_matrix



test_vector=np.zeros(shape=(100,len(corpus)))

for j,i in enumerate(x_test[0:100]):
    for some in tokenizer.tokenize(i[0]) :
        if some not in stopwords:
            test_vector[j][mp[some]]=1


y_test=y_test=="REAL"
y_testx=np.zeros(shape=(y_test.shape[0]))
for i,d in enumerate(y_train):
    if d :
        y_testx[i]=1
    
y_test=y_testx


y_pred=model.predict(test_vector)
cm=confusion_matrix(y_test[0:100],y_pred)
