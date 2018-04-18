import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self,session,input_size,output_size,name = 'name'):
        self.session = session
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self. Network_config()

    def Network_config(self):
        with tf.variable_scope(self.name):
            H_size = 10
            self.X = tf.placeholder(shape=[None,self.input_size], dtype=tf.float32, name='input_X')
            self.W1 = tf.get_variable("W1",[self.input_size, H_size],initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable("B1",[H_size],initializer=tf.contrib.layers.xavier_initializer())
            self.X2 = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)

            self.W2 = tf.get_variable("W2",[H_size,H_size],initializer=tf.contrib.layers.xavier_initializer())
            self.b2 = tf.get_variable("B2",[H_size],initializer=tf.contrib.layers.xavier_initializer())
            self.X3= tf.nn.relu(tf.matmul(self.X2, self.W2) + self.b2)

            self.W3 = tf.get_variable("W3",[H_size, self.output_size],initializer=tf.contrib.layers.xavier_initializer())
            self.b3 = tf.get_variable("B3",[self.output_size],initializer=tf.contrib.layers.xavier_initializer())
            self.Qpred = tf.matmul(self.X3, self.W3) + self.b3

        self.Y = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32,name='output_Y')
        self.Loss = tf.reduce_mean(tf.square(self.Y-self.Qpred))
        self.train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.Loss)

    def predict(self,state):
        X = np.reshape(state,[1,self.input_size])
        return self.session.run(self.Qpred,feed_dict = {self.X : X})

    def update(self,X_stack,Y_stack):
        return self.session.run([self.Loss,self.train],feed_dict = {self.X : X_stack, self.Y : Y_stack})
