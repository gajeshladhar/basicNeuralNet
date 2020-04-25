import os
import numpy as np
labels=[]
texts=[]
imdb_dir="aclImdb"
train_dir=os.path.join(imdb_dir,'train')

for i,label in enumerate(['neg','pos']):
    dir_name=os.path.join(train_dir,label)
    for fname in os.listdir(dir_name):
        if fname[-4:]=='.txt':
            f=open(os.path.join(dir_name,fname))
            texts.append(f.read())
            labels.append(i)

maxlen=100
samples=200
validation_samples=10000
max_words=10000

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer=Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)
word_index=tokenizer.word_index

data=pad_sequences(sequences,maxlen=maxlen)
labels=np.array(labels)

x=range(25000)
np.random.shuffle(x)

train_data=np.array(data[x[:samples]],dtype=int)
train_labels=labels[x[:samples]]
val_data=data[x[samples:samples+validation_samples]]
val_labels=labels[x[samples:samples+validation_samples]]

embedding_index={}
fp=open("glove.6B.100d.txt")

for line in fp:
    values=line.split()
    word=values[0]
    coef=np.array(values[1:])
    embedding_index[word]=coef
    
dims=100
matrix=np.zeros(shape=(max_words,dims))

for a,b in word_index.items() :
    if b<max_words :
        vect=embedding_index.get(a)
        if vect is not None :
            matrix[b]=embedding_index[a]
    

from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense

Network=Sequential()
Network.add(Embedding(10000,100,input_length=100))
Network.add(Flatten())
Network.add(Dense(32,activation="relu"))
Network.add(Dense(1,activation="sigmoid"))

Network.summary()

Network.layers[0].set_weights([matrix])
Network.layers[0].trainable=False

Network.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["acc"])

hist=Network.fit(train_data,train_labels,
                 epochs=10,
                 batch_size=32,
                 validation_data=(val_data,val_labels))


result=hist.history

import matplotlib.pyplot as plt

plt.plot(range(1,11),result['val_acc'],color="red")
plt.plot(range(1,11),result['acc'])
plt.xlabel("No. of Epochs ")
plt.ylabel("  Accuracy ")
plt.show()






















