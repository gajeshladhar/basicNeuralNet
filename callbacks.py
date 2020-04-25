import keras
callbacks_list=[
        keras.callbacks.EarlyStopping(
                monitor="acc",
                patience=15),
                keras.callbacks.ModelCheckpoint(
                        filepath="xyzzz",
                        monitor="val_loss",
                        save_best_only=True),
                        keras.callbacks.ReduceLROnPlateau(
                                monitor="val_loss",factor=0.1,patience=10)]
                
        

#Own CallBacks....

import numpy as np

class OWNCALLBACKS(keras.callbacks.Callback):
    def set_model(self,model):
        self.model=model
        layers_outputs=[layer.output for layer in model.layers]
        self.activation_model=keras.models.Model(model.input,layers_outputs)
        
    def on_epoch_end(self,epoch,logs=None):
        validation_sample=np.array(self.validation_data[:][0])
        print(validation_sample.shape)
        activations=self.model.predict(validation_sample)
        f=open('xyz'+str(epoch)+'.npz',"w")
        np.savez(f,activations)
        



input_data=keras.Input(shape=(10,))
x=keras.layers.Dense(10,activation="relu")(input_data)
x=keras.layers.Dense(1,activation="sigmoid")(x)
model=keras.Model(input_data,x)
model.compile(optimizer="rmsprop",loss="mae",metrics=["acc"])
layers_outputs=[layer.output for layer in model.layers]



data=np.random.random(size=(1000,10))
answer=np.zeros(shape=(1000,1))
for i in range(1000):
    answer[i]=1.0/(1+np.exp(sum(data[i])))


model.fit(data,answer,validation_split=0.5,epochs=10,callbacks=[OWNCALLBACKS()])
