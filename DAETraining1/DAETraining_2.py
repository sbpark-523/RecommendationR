import tensorflow as tf
import numpy as np
import MyConfiguration as myCfg
import datetime
import DAETraining1.NoiseMask as NoiseMask
import matplotlib.pyplot as plt

def numpy_to_2tensor(ndarray1, ndarray2):
    tensor1 = tf.convert_to_tensor(ndarray1, dtype=tf.float32)
    tensor2 = tf.convert_to_tensor(ndarray2, dtype=tf.float32)
    return tensor1, tensor2

learning_rate =0.01

training_epoch = 1000
batch_size = 200
total_size = 7200 # 모름

myDecay = False
stepDecay = False

# total epoch
global_step = tf.Variable(0, trainable=False)

X = tf.placeholder(tf.float32, [None, 2048], name='X_noisy')   # [batch_size, input of length, dimension]
Y = tf.placeholder(tf.float32, [None, 2048], name='Y_clean')   # [batch_size, input of length, dimension]
if myDecay:
    m_learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
elif stepDecay:
    m_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=200, decay_rate=0.96, staircase=True)
else:
    m_learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

X_reshape = tf.reshape(X, (-1, 2048, 1))
Y_reshape = tf.reshape(Y, (-1, 2048, 1))

# width (size of feature map), input channel, output channel
E_w1 = tf.Variable(tf.random_normal([3, 1, 4], stddev=0.01), name='Ew1')
E_w2 = tf.Variable(tf.random_normal([5, 4, 8], stddev=0.01), name='Ew2')
E_w3 = tf.Variable(tf.random_normal([5, 8, 16], stddev=0.01), name='Ew3')

D_w1 = tf.Variable(tf.random_normal([5, 8, 16], stddev=0.01), name='Dw1')
D_w2 = tf.Variable(tf.random_normal([5, 4, 8], stddev=0.01), name='Dw2')
D_w3 = tf.Variable(tf.random_normal([3, 1, 4], stddev=0.01), name='Dw3')


# input shape X_reshape --> (batch size, 2048, 1)
def encoder(X):
    EC1 = tf.nn.conv1d(value=X, filters=E_w1, stride=1, padding='SAME', name='E/conv1')
    EC1 = tf.nn.leaky_relu(EC1, 0.01)
    print(EC1)

    EC2 = tf.nn.conv1d(value=EC1, filters=E_w2, stride=1, padding='SAME', name='E/conv2')
    EC2 = tf.nn.leaky_relu(EC2, 0.01)
    print(EC2)

    EC3 = tf.nn.conv1d(value=EC2, filters=E_w3, stride=1, padding='SAME', name='E/conv3')
    EC3 = tf.nn.leaky_relu(EC3, 0.01)
    print(EC3)

    output = EC3

    return output

def decoder(mid):

    mid_size = tf.shape(mid)[1]
    print(mid_size*2)
    DC1 = tf.contrib.nn.conv1d_transpose(value=mid, filter=D_w1, output_shape=[batch_size, mid_size, 8], stride=1, padding='SAME', name='D/conv1', data_format='NWC')
    DC1 = tf.nn.leaky_relu(DC1, 0.01)
    print(DC1)

    DC2 = tf.contrib.nn.conv1d_transpose(value=DC1, filter=D_w2, output_shape=[batch_size, mid_size, 4], stride=1, padding='SAME', name='D/conv2', data_format='NWC')
    DC2 = tf.nn.leaky_relu(DC2, 0.01)
    print(DC2)

    DC_final = tf.contrib.nn.conv1d_transpose(value=DC2, filter=D_w3, output_shape=[batch_size, mid_size, 1], stride=1, padding='SAME', name='D/conv_final')
    print(DC_final)

    return DC_final

encoded_X = encoder(X_reshape)
decoded_X = decoder(encoded_X)
print('decoded shape:',decoded_X.shape)


cost = tf.reduce_mean(tf.pow(Y_reshape - decoded_X, 2))
print('cost shape:',cost.shape)
optimizer = tf.train.AdamOptimizer(learning_rate=m_learning_rate).minimize(cost, global_step=global_step)


def test(denoised, wo_noise):
    return tf.sqrt(tf.reduce_sum(tf.squared_difference(denoised, wo_noise)))

test_model = test(decoded_X, Y_reshape)

### Saver
param_list = [E_w1, E_w2, E_w3, D_w1, D_w2, D_w3]
saver = tf.train.Saver(param_list)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

s = datetime.datetime.now()
print("Learning start at {}".format(s))

noiseMask = NoiseMask.NoiseMask(myCfg.train_data_route)
xTrain, yTrain = noiseMask.createTrainingData()

last_dist = 1
validation_list = []

for step in range(training_epoch):
    train_loss = 0
    for loop in range(int(total_size/batch_size)):
        loss1 = 0.0
        x_batch = xTrain[int(loop*batch_size): int((loop+1)*batch_size)]
        y_batch = yTrain[int(loop*batch_size): int((loop+1)*batch_size)]
        # x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        if myDecay:
            _, loss1 = sess.run([optimizer, cost], feed_dict={X: x_batch, Y: y_batch, m_learning_rate: learning_rate})
        elif stepDecay:
            _, loss1 = sess.run([optimizer, cost], feed_dict={X: x_batch, Y: y_batch})
        else:
            _, loss1 = sess.run([optimizer, cost], feed_dict={X: x_batch, Y: y_batch, m_learning_rate: learning_rate})

        train_loss += loss1/(int(total_size/batch_size))

    if step % 10 == 0:
        s = datetime.datetime.now()
        print('{}\tStep = {}, Cost = {}'.format(s, step, train_loss))

    if step % 50 == 0:
        xValid, yValid = noiseMask.createValidationData()
        now_dist = 0
        print('data... OK')
        ############## draw graph #############
        denoised_size = 2
        dn = sess.run(decoded_X, feed_dict={X: xValid[0:200]})
        fig, ax = plt.subplots(denoised_size+1, 1, figsize=(20, denoised_size*3))
        ax[0].plot(np.reshape(xValid[0], (2048, 1)))    # noisy
        ax[0].set_ylim((0, 7))
        ax[0].grid()
        ax[1].plot(np.reshape(dn[0], (2048, 1)))        # denoised
        ax[1].set_ylim((0, 7))
        ax[1].grid()
        ax[2].plot(np.reshape(yValid[0], (2048, 1)))    # original
        ax[2].set_ylim((0, 7))
        ax[2].grid()

        plt.savefig('check_TS_adam_2/{}.png'.format(str(step).zfill(3)), bbox_inches='tight')


        for vali_loop in range(int(2400/batch_size)):
            acc_val = 0.0
            x_vali_batch = xValid[int(vali_loop*batch_size): int((vali_loop+1)*batch_size)]
            y_vali_batch = yValid[int(vali_loop*batch_size): int((vali_loop+1)*batch_size)]
            if myDecay:
                acc_val = sess.run(test_model, feed_dict={X: x_vali_batch, Y: y_vali_batch, m_learning_rate: learning_rate})
            elif stepDecay:
                acc_val = sess.run(test_model, feed_dict={X: x_vali_batch, Y: y_vali_batch, global_step: step})
            else:
                acc_val = sess.run(test_model, feed_dict={X: x_vali_batch, Y: y_vali_batch, m_learning_rate: learning_rate})
            now_dist += acc_val / (int(2400/batch_size))
        if myDecay:
            if len(validation_list) < 5:
                vali_rate = now_dist/last_dist
                validation_list.append(vali_rate)
            if len(validation_list) ==5:
                count = 0
                for v in validation_list:
                    if v<1:
                        count += 1
                if count >= 3:
                    learning_rate *= 0.5
                    validation_list.clear()
                    print('--------learning rate decayed!!')
                else:
                    validation_list.pop(0)
        if myDecay or stepDecay:
            print('Distance: {}, learning rate: {}'.format(now_dist, m_learning_rate.eval()))
        else:
            print('Distance: {}, learning rate: {}'.format(now_dist, learning_rate))
        last_dist = now_dist

        save_path = saver.save(sess, "./model/model_12(lr_01_e1000_adam)/model.ckpt")
        print("Saved in : {}".format(save_path))

    xTrain, yTrain = noiseMask.createTrainingData()