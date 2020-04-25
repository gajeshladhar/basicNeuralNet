from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

max_features=10000
maxlen=500
batch_size=32

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)

np.load = np_load_old

x_train=sequence.pad_sequences(x_train,maxlen)
x_test=sequence.pad_sequences(x_test,maxlen)

x_train[0]

from keras.layers import SimpleRNN,Embedding,Dense
from keras.models import Sequential

model = Sequential()
model.add(Embedding(10000,32))
#model.summary()

model.add(SimpleRNN(32,return_sequences=True))
model.add(SimpleRNN(32,return_sequences=True))
model.add(SimpleRNN(32,return_sequences=True))
model.add(SimpleRNN(32))

model.add(Dense(units=1,activation="sigmoid"))

model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["acc"])


history=model.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.2)



import matplotlib.pyplot as plt

acc=history.history["acc"]
val_acc=history.history["val_acc"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,label="Accuracy",color="blue")
plt.plot(epochs,val_acc,label="Val Accuracy",color="red")
plt.xlabel("Epochs")
plt.legend()
plt.show()
















