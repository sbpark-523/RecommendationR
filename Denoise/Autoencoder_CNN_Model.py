import tensorflow as tf
import numpy as np
import MyConfiguration as myCfg
import matplotlib.pyplot as plt
import os

# X = tf.placeholder(tf.float32, [None, None, 1])   # [batch_size, input of length, dimension]

# width (size of feature map), input channel, output channel
E_w1 = tf.Variable(tf.random_normal([3, 1, 8], stddev=0.01), name='Ew1')
E_w2 = tf.Variable(tf.random_normal([5, 8, 16], stddev=0.01), name='Ew2')

D_w1 = tf.Variable(tf.random_normal([5, 16, 16], stddev=0.01), name='Dw1')
D_w2 = tf.Variable(tf.random_normal([3, 8, 16], stddev=0.01), name='Dw2')
D_w3 = tf.Variable(tf.random_normal([3, 8, 1], stddev=0.01), name='Dw3')

class Model:
    def __init__(self, sess):
        self.sess = sess

    def encoder(self, X):
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

    def decoder(self, mid):
        mid_size = tf.shape(mid)[1]
        # print(mid_size*2)
        DC1 = tf.contrib.nn.conv1d_transpose(value=mid, filter=D_w1, output_shape=[1, mid_size*2, 16], stride=2, padding='SAME', name='D/conv1', data_format='NWC')
        DC1 = tf.nn.leaky_relu(DC1, 0.01)
        print(DC1)

        DC2 = tf.contrib.nn.conv1d_transpose(value=DC1, filter=D_w2, output_shape=[1, mid_size*4, 8], stride=2, padding='SAME', name='D/conv2', data_format='NWC')
        DC2 = tf.nn.leaky_relu(DC2, 0.01)
        print(DC2)

        DC_final = tf.nn.conv1d(value=DC2, filters=D_w3, stride=1, padding='SAME', name='D/conv_final')
        DC_final = tf.nn.leaky_relu(DC_final, 0.01)
        print(DC_final)

        return DC_final

class DAE:

    def __init__(self):
        # model_path = "D:/Python/git/RecommendationR/Denoise/model/model_7(start_lr_01_e2000_adam)"
        # model_path = "D:/Python/git/RecommendationR/Denoise/model/model_7(start_lr_01_e2000_adam)/"
        abspath = os.path.abspath("./").replace('\\', '/')
        self.X = tf.placeholder(tf.float32, [1, 2048], name='X_noisy')   # [batch_size, input of length, dimension]
        self.Y = tf.placeholder(tf.float32, [1, 2048], name='Y_clean')   # [batch_size, input of length, dimension]

        self.X_reshape = tf.reshape(self.X, (-1, 2048, 1))
        self.Y_reshape = tf.reshape(self.Y, (-1, 2048, 1))
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        # self.saver = tf.train.import_meta_graph(model_path+"/model.ckpt.meta")
        # load model
        self.saver = tf.train.Saver()
        print(abspath+"/"+myCfg.trained_model_route)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(abspath+"/"+myCfg.trained_model_route+"/"))

        self.daeModel = Model(self.sess)

        self.encoded_X = self.daeModel.encoder(self.X_reshape)
        self.decoded_X = self.daeModel.decoder(self.encoded_X)
        print('decoded shape:',self.decoded_X.shape)
        self.test_model = self.test(self.decoded_X, self.Y_reshape)
        print(E_w1.eval())

        # cost = tf.reduce_mean(tf.pow(X - self.decoded_X, 2))
        # print('cost shape:',cost.shape)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    def denoise(self, noisy_data, ori):
        result = self.sess.run([self.decoded_X], feed_dict={self.X: [noisy_data]})
        dists = self.sess.run(self.test_model, feed_dict={self.X: [noisy_data], self.Y: [ori]})
        print("Distance: ", dists)

        # fig, ax = plt.subplots(3, 1, figsize=(20, 3*3))
        # ax[0].plot(np.reshape(noisy_data, (2048, 1)))
        # # ax[0].set_ylim((0, 6.5))
        # ax[0].grid()
        # ax[1].plot(np.reshape(result, (2048, 1)))
        # # ax[1].set_ylim((0, 6.5))
        # ax[1].grid()
        # ax[2].plot(np.reshape(ori, (2048, 1)))
        # # ax[2].set_ylim((0, 6.5))
        # ax[2].grid()

        # plt.savefig('./{}.png'.format(str(1).zfill(3)), bbox_inches='tight')
        result = np.reshape(result, (2048))
        return result[:myCfg.window_length]


    def test(self, denoised, wo_noise):
        return tf.sqrt(tf.reduce_sum(tf.squared_difference(denoised, wo_noise)))

