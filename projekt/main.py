#Program ma za zadanie nauczyc siec neuronowa z uzyciem techniki convultion
#rozponawac liczby na obrazkach ze zbioru MNIST

#import biblioteki tensorflow
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#import danych MINIST
mnist = input_data.read_data_sets('MNIST_data', one_hot=True )

#utworzenie interaktywnej sesji
sess = tf.InteractiveSession()

#utworzenie zmiennych:
# x - wejście do sieci neuronowej - 784 warosci
# y_ - wyjscie z sieci - 10 wartości okreslajacych wykrytą liczbe
xInput = tf.placeholder(tf.float32, shape=[None, 784])
Output = tf.placeholder(tf.float32, shape=[None, 10])

# utworzenie zmiennych tzw bias
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#zainicjalizowanie zmiennych w bibliotece TF
sess.run(tf.global_variables_initializer())
# ustawienie równania 
y = tf.matmul(xInput,W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmaxInput_cross_entropy_with_logits(y, Output))
#ustawienie kroku, z wartością naucznia na 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={xInput: batch[0], Output: batch[1]})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(Output,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32) )
print(accuracy.eval(feed_dict={xInput: mnist.test.images, y_: mnist.test.labels}))
#program with convulution
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(xInput, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#ustawienia równania dla y z convultion
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer
#uczenie 
for i in range(20000):
#pobranie danych ze zbioru MINST
  batch = mnist.train.next_batch(50)
  #wyświetlenie informacji co jakiś czas o skutkach uczenia
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("krok: %d,  accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# dane testwoe
print("%g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))