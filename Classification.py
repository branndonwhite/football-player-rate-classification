from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
import pandas as pd
import numpy as np

class Classification:
    def __init__(self, input_node, hidden_1_node, hidden_2_node, output_node, learning_rate):
        super().__init__()
        
        self.weight = {
            'to_hidden1' : tf.Variable(tf.random_normal([input_node, hidden_1_node])),
            'to_hidden2' : tf.Variable(tf.random_normal([hidden_1_node, hidden_2_node])),
            'to_output' : tf.Variable(tf.random_normal([hidden_2_node, output_node]))
        }

        self.bias = {
            'to_hidden1' : tf.Variable(tf.random_normal([hidden_1_node])),
            'to_hidden2' : tf.Variable(tf.random_normal([hidden_2_node])),
            'to_output' : tf.Variable(tf.random_normal([output_node]))
        }

        self.x = tf.placeholder(tf.float32, [None, input_node])
        self.target = tf.placeholder(tf.float32, [None, output_node])

        self.prediction = self.predict()
        self.loss = tf.reduce_mean(.5 * (self.target - self.prediction) ** 2)
        self.train = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

    def predict(self):
        wx_b1 = tf.matmul(self.x, self.weight['to_hidden1']) + self.bias['to_hidden1']
        y1 = tf.nn.tanh(wx_b1)

        wx_b2 = tf.matmul(y1, self.weight['to_hidden2']) + self.bias['to_hidden2']
        y2 = tf.nn.tanh(wx_b2)

        wx_b3 = tf.matmul(y2, self.weight['to_output']) + self.bias['to_output']
        output = tf.nn.sigmoid(wx_b3)

        return output

    def train_and_evaluate(self, x_train, x_test, x_validate, y_train, y_test, y_validate, epoch=5000):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(max_to_keep=10)
            error_temp = .1

            for i in range(epoch+1):
                sess.run(self.train, feed_dict={self.x : x_train, self.target : y_train})

                if i % 100 == 0:
                    print('\n\nCurrent Epoch: {} Current Error: {}%'.format(i, sess.run(self.loss, feed_dict={self.x : x_train, self.target : y_train}) * 100))
                    
                    prediction = tf.equal(tf.argmax(self.target, axis=1), tf.argmax(self.prediction, axis=1))

                    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
                    print('Accuracy: {}%'.format(sess.run(accuracy, feed_dict = {self.x : x_test, self.target : y_test}) * 100))

                if i % 500 == 0:
                    error = sess.run(self.loss, feed_dict={self.x : x_validate, self.target : y_validate}) * 100
                    if i == 500:
                        error_temp = error
                        # print('Current Validation Error: {}%'.format(error_temp))
                        saver.save(sess, 'bpnn-model/validation_model.ckpt', global_step=i)
                    else:
                        if(error < error_temp):
                            # print('Current Validation Error: {}%'.format(error))
                            # print('Previous Validation Error: {}%'.format(error_temp))
                            error_temp = error
                            saver.save(sess, 'bpnn-model/validation_model.ckpt', global_step=i)

data = pd.read_csv("O202-COMP7117-VJ03-00-classification.csv")

data['Tackle'] = (data['StandingTackle'] + data['SlidingTackle']) / 2
data['GK'] =  (data['GKDiving'] + data['GKHandling'] + data['GKKicking'] + data['GKPositioning'] + data['GKReflexes']) / 5

feature = data[['International Reputation', 'ShortPassing', 'LongPassing', 'BallControl', 'Reactions', 'Penalties', 'Tackle', 'GK']]
target = data[['Overall']]

scaler = MinMaxScaler()
feature = scaler.fit_transform(feature)

encoder = OneHotEncoder(sparse=False)
target = encoder.fit_transform(target)

mean = tf.reduce_mean(feature, axis=0)
centered_feature = feature - mean

with tf.Session() as sess:
    dataset = sess.run(centered_feature)

    pca = PCA(n_components=5)
    new_dataset = pca.fit_transform(dataset)

x_train, x_test, y_train, y_test = train_test_split(new_dataset, target, test_size=0.3, shuffle=True)

x_validate, _, y_validate, _ = train_test_split(x_train, y_train, train_size=0.2)

input_layer = 5
hidden_layer_1 = 6
hidden_layer_2 = 8
output_layer = 5
learning_rate = .2

bpnn = Classification(input_layer, hidden_layer_1, hidden_layer_2, output_layer, learning_rate)
bpnn.train_and_evaluate(x_train, x_test, x_validate, y_train, y_test, y_validate)