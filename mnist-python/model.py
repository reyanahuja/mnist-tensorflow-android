import tensorflow as tf
#we umport the mnist data from the tensorflow examples
from tensorflow.examples.tutorials.mnist import input_data
#use predefined input data sets function of the input data to load the data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
#We define some tensorflow placeholders
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name = 'x_input')
W = tf.Variable(initial_value=tf.zeros(shape=[784,10]), name = 'W')
b = tf.Variable(initial_value=tf.zeros(shape=[10]), name = 'b')
#Note that the output will be an array of size 10
y_actual = tf.add(x=tf.matmul(a=x_input, b=W, name='matmul'),y=b,name='y_actual')
y_expected = tf.placeholder(dtype = tf.float32, shape=[None,10], name = 'y_expected')
#define crossentropy loss
cel = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=y_expected,logits=y_actual), name = 'cel')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5, name = 'optimizer')

train_step = optimizer.minimize(loss=cel, name='train_step')

saver = tf.train.Saver()

#create a tensorflow session
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

tf.train.write_graph(graph_or_graph_def=session.graph_def,
                     logdir='.',
                     name='mmistmodel.pbtxt',
                     as_text=False)
#We now train the model
for _ in range(1000):
    batch=mnist_data.train.next_batch(100)
    train_step.run(feed_dict={x_input:batch[0], y_expected: batch[1]})
#after training, we save the model for android import
saver.save(sess=session,save_path='mnistmodel.ckpt')


#This part includes evaluation of the model
corr= tf.equal(x=tf.argmax(y_actual,1),y=tf.argmax(y_expected,1))
ac = tf.reduce_mean(tf.cast(x=corr, dtype=tf.float32))
print(ac.eval(feed_dict={x_input: mnist_data.test.images, y_expected: mnist_data.test.labels}))

print(session.run(fetches=y_actual, feed_dict={x_input: [mnist_data.test.images[0]]}))
print(mnist_data.test.images[0])

