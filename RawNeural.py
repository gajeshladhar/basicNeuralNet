import tensorflow as tf
from tensorflow.contrib.layers import fully_connected,batch_norm
import numpy as np

n_inputs=28*28
n_hidden1=200
n_hidden2=100
n_outputs=10

X=tf.placeholder("float32",shape=(None,n_inputs),name="X")
Y=tf.placeholder("int64",shape=(None),name="Y")



def Neuron_layer(X,nodes,activation=None):
    n_inputs=int(X.get_shape()[1])
    stddev=2/np.sqrt(n_inputs)
    init=tf.truncated_normal((n_inputs,nodes),stddev=stddev)
    W=tf.Variable(init,dtype="float32")
    b=tf.Variable(np.random.random(nodes),dtype="float32")
    
    z=tf.matmul(X,W,)+b
    # Effort to Implement Batch Normalization....
    if(1):
        Alpha=tf.Variable(0.4,dtype="float32")
        Beta=tf.Variable(0.4,dtype="float32")
        new_z=Alpha*((z-tf.reduce_mean(z))/tf.sqrt(tf.square((z-tf.reduce_mean(z)))/50))+Beta
    else :
        pass
       # new_z=Alpha*((z-np.mean(z,dtype="float32"),axis=1)/np.std(z,dtype="float32",axis=1))+Beta
        
    if(activation=="relu"):
        return tf.nn.sigmoid(new_z)
    elif(activation=="softmax") :
        return tf.nn.softmax(new_z)


#Layers....
is_training=tf.placeholder(dtype=tf.bool,shape=())
bn={
    'is_training':is_training,
    'decay':0.99,
    'updates_collections':None,
    'scale': True
    }

       
with tf.name_scope("Layersx") :
    with tf.contrib.framework.arg_scope(
            [fully_connected],normalizer_fn=batch_norm,normalizer_params=bn,
            weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.01)) :
        hidden1=fully_connected(X,n_hidden1,activation_fn=tf.nn.sigmoid,scope="hidden1x")
        hidden2=fully_connected(hidden1,n_hidden2,activation_fn=tf.nn.sigmoid,scope="hidden2x")
        logits=fully_connected(hidden2,n_outputs,activation_fn=tf.nn.softmax,scope="logitsx")
        
        

    
with tf.name_scope("Loss"):
    entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits)
    base_loss=tf.reduce_mean(entropy)
    reg_loss=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss=tf.add([base_loss],reg_loss)
    

with tf.name_scope("Optimizer"):
    optimizer=tf.train.AdamOptimizer(learning_rate=0.03,)
    op=optimizer.minimize(loss)
    
    
    
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,Y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    

init=tf.global_variables_initializer()
saver=tf.train.Saver()
#Execution....
    
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/")
    
batch_size=50
n_epochs=400

with tf.Session() as session :
    session.run(init)
    for epoch in range(n_epochs):
        print("\nEpoch : "+str(epoch)+"/"+str(n_epochs))
        for batch in range(mnist.train.num_examples // batch_size):
            X_batch,Y_batch=mnist.train.next_batch(batch_size)
            session.run(op,feed_dict={X:X_batch,Y:Y_batch,is_training:True})
        acc_train=accuracy.eval(feed_dict={X:X_batch,Y:Y_batch,is_training:False})
        acc_test=accuracy.eval(feed_dict={X:mnist.test.images,Y:mnist.test.labels,is_training:False})
        
        print("Training Acc : "+str(acc_train))
        print("Testing Acc : "+str(acc_test))
        saver.save(session,"./mnist_model.h5")

        
with tf.Session() as session :
    saver.restore(session,"./mnist_model.h5")
    
    #acc_test=accuracy.eval(feed_dict={X:mnist.test.images,Y:mnist.test.labels})
    print(np.argmax(session.run(logits,feed_dict={X:mnist.test.images[0:1]})))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    