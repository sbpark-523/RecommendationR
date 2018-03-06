import tensorflow as tf
import numpy as np

learning_rate =0.001

training_epoch = 20
batch_size = 1
total_size = 10000 # 모름

INPUT_SIZE = 128
# 32, 64, 128

test_input = np.array([[float(x) for x in range(128)]])
test_input = test_input.reshape(-1, INPUT_SIZE, 1)
print(test_input)
print(test_input.shape)

X = tf.placeholder(tf.float32, [None, None, 1])   # [batch_size, input of length, dimension]

# width (size of feature map), input channel, output channel
E_w1 = tf.Variable(tf.random_normal([3, 1, 8], stddev=0.01))
E_w2 = tf.Variable(tf.random_normal([5, 8, 16], stddev=0.01))

D_w1 = tf.Variable(tf.random_normal([5, 16, 16], stddev=0.01))
D_w2 = tf.Variable(tf.random_normal([3, 8, 16], stddev=0.01))
D_w3 = tf.Variable(tf.random_normal([3, 8, 1], stddev=0.01))


# D_w1 = tf.Variable(tf.random_normal([5, 16, 8], stddev=0.01))
# D_w2 = tf.Variable(tf.random_normal([3, 1, 3], stddev=0.01))


def encoder(X):
    EC1 = tf.nn.conv1d(value=X, filters=E_w1, stride=1, padding='SAME', name='E/conv1')
    EC1 = tf.nn.leaky_relu(EC1, 0.01)
    print(EC1)

    EP1 = tf.layers.max_pooling1d(inputs=EC1, pool_size=2, padding='SAME', strides=2)
    print(EP1)

    EC2 = tf.nn.conv1d(value=EP1, filters=E_w2, stride=1, padding='SAME', name='E/conv2')
    EC2 = tf.nn.leaky_relu(EC2, 0.01)
    print(EC2)

    EP2 = tf.layers.max_pooling1d(inputs=EC2, pool_size=2, padding='SAME', strides=2)
    print(EP2)

    output = EP2

    return output

def decoder(mid):
    DC1 = tf.contrib.nn.conv1d_transpose(value=mid, filter=D_w1, output_shape=[1, 64, 16], stride=2, padding='SAME', name='D/conv1', data_format='NWC')
    DC1 = tf.nn.leaky_relu(DC1, 0.01)
    print(DC1)

    DC2 = tf.contrib.nn.conv1d_transpose(value=DC1, filter=D_w2, output_shape=[1, 128, 8], stride=2, padding='SAME', name='D/conv2', data_format='NWC')
    DC2 = tf.nn.leaky_relu(DC2, 0.01)
    print(DC2)

    DC_final = tf.nn.conv1d(value=DC2, filters=D_w3, stride=1, padding='SAME', name='D/conv_final')
    DC_final = tf.nn.leaky_relu(DC_final, 0.01)
    print(DC_final)

    return DC_final

encoded_X = encoder(X)
decoded_X = decoder(encoded_X)
print('decoded shape:',decoded_X.shape)


cost = tf.reduce_mean(tf.pow(X - decoded_X, 2))
print('cost shape:',cost.shape)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: test_input})

    if step % 100 == 0:
        print('Step = {}, Cost = {}'.format(step, loss))


# for epoch in range(training_epoch):
#     avg_loss = 0
#
#     total_batch = int(total_size / batch_size)
#     for i in range(total_batch):
#         batch_xs = () """------------ 여기 수정해야함 """
#         _, loss_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})
#
#         avg_loss += loss_val / total_batch