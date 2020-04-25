import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
from hyperas import optim
import hyperas
import hyperopt
from keras.datasets import mnist

def data():
    
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
   
    return X_train[:5000], Y_train[:5000], X_test[:1000], Y_test[:1000]


def model(X_train, Y_train, X_test, Y_test):
   
    model = layers.Sequential()
    model.add(layers.Dense(512, input_shape=(784,)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout({{hyperopt.uniform(0, 1)}}))
    model.add(layers.Dense({{hyperopt.choice([256, 512, 1024])}}))
    model.add(layers.Activation({{hyperopt.choice(['relu', 'sigmoid'])}}))
    model.add(layers.Dropout({{hyperopt.uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if {{hyperopt.choice(['three', 'four'])}} == 'four':
        model.add(layers.Dense(100))
        model.add({{hyperopt.choice([layers.Dropout(0.5), layers.Activation('linear')])}})
        model.add(layers.Activation('relu'))

    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer={{hyperopt.choice(['rmsprop', 'adam', 'sgd'])}},
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size={{hyperopt.choice([64, 128])}},
              nb_epoch=1,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': hyperopt.STATUS_OK, 'model': model}



best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          max_evals=10,
                                          algo=hyperopt.rand.suggest,
                                          notebook_name='name', # This is important!
                                          trials=hyperopt.Trials())



