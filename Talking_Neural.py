

import keras

import numpy as np

path=keras.utils.get_file('data.txt',origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")

text=open(path).read().lower()

len(text)

maxlen=60
step=3
sentences=[]
next_chars=[]

for i in range(0,len(text)-maxlen,step):
  sentences.append(text[i:(i+maxlen)])
  next_chars.append(text[i+maxlen])

len(sentences)

chars=sorted(list(set(text)))

char_indices=dict((char,chars.index(char)) for char in chars)

x=np.zeros(shape=(len(sentences),maxlen,len(chars)),dtype=np.bool)
y=np.zeros(shape=(len(sentences),len(chars)),dtype=np.bool)

for i,s in enumerate(sentences):
  for k,char in enumerate(s):
    x[i,k,char_indices[char]]=1
  y[i,char_indices[next_chars[i]]]=1

import tensorflow as tf
input_layer=tf.keras.Input(shape=(maxlen,len(chars)))
lstm=tf.keras.layers.LSTM(128)(input_layer)
norm=tf.keras.layers.BatchNormalization()(lstm)
dense_1=tf.keras.layers.Dense(32,activation="relu")(norm)
norm_1=tf.keras.layers.BatchNormalization()(dense_1)
dense=tf.keras.layers.Dense(len(chars),activation="softmax")(norm_1)

model=tf.keras.models.Model(input_layer,dense)
model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["acc"])

def sample(preds,temperature=1.0):
  preds=np.asarray(preds).astype("float")
  preds=np.log(preds)/temperature
  exp_pred=np.exp(preds)
  preds=exp_pred/np.sum(exp_pred)
  multi=np.random.multinomial(1,preds,1)
  return np.argmax(multi) #Returning Index

i=np.random.randint(0,len(text)-maxlen-10)
init=text[i:i+maxlen]

model.summary()

model.fit(x,y,batch_size=128,epochs=10)

msg=" "
x_test=np.zeros(shape=(1,maxlen,len(chars)))

for i in range(0,200):
  x_test=np.zeros(shape=(1,maxlen,len(chars)))
  for k,char in enumerate(init):
    x_test[0,k,char_indices[char]]=1
  
  y_pred=model.predict(x_test)[0]
  index=sample(y_pred,0.5)
  #print(chars[index],end="")
  msg+=(chars[index])
  init+=str(chars[index])
  init=init[1:]

msg
