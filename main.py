import random
import numpy as np
import math

import MyConfiguration as myCfg
import Denoise.MovingAverage as ma
import Denoise.WaveletTransform as wt
import DistanceCalculator
import KalmanFilter
import Denoise.Autoencoder_CNN_Model as DAE

data = np.loadtxt(myCfg.data_route, delimiter=',', dtype=np.float32)

print("Load data from:", myCfg.data_route, data.shape)
# print(data)

result_file = open('result_file/myresult_32.csv', 'w')
noise_file = open('./noise_file_wt.csv','w')

reshaped_data = np.transpose(data)
windows = int(len(reshaped_data[0])/myCfg.window_length)
kalman_filtering = KalmanFilter.KalmanFilter(result_file, True)

# load model
if myCfg.method == "DAE":
    denoisingAutoencoder = DAE.DAE()

dist = DistanceCalculator.DistanceCalculator(noise_file)

result = []

quarter = int(myCfg.window_length/4)
half = int(myCfg.window_length/2)

total_dist = 0
ori_dist = 0
R_sum = 0
print("##########################################")
print("Window size: {}, total_window: {}".format(myCfg.window_length, myCfg.total_windows))
print("##########################################")
print("==== Method: {}".format(myCfg.method))
for x in range(2):
# for x in range(myCfg.total_windows):
    ori_data = reshaped_data[myCfg.original_row, int(x * myCfg.window_length): int((x+1) * myCfg.window_length)]
    batch_data = reshaped_data[myCfg.noisy_row, int(x * myCfg.window_length): int((x+1) * myCfg.window_length)]

    if x > 0:
        result, batch_dist = kalman_filtering.filter(sensor=batch_data, clean_data=ori_data)
        total_dist += batch_dist
        # print('Filtered: {}'.format(result))


    if myCfg.method == "MA":
        """ 1. Moving Average """
        ma.sensor_data = batch_data
        denoised_data = ma.MovingAverage()
        for va in denoised_data:
            print(va)
    elif myCfg.method == "WT":
        """ 2. Wavelet transform """
        waveletTransform = wt.WaveletTransform(myCfg.window_length)
        transformed = waveletTransform.Daubechies_forward_FWT_1d(batch_data)
        removed = np.concatenate((transformed[:quarter], np.zeros(shape=(len(transformed)-quarter))), axis=0)
        denoised_data = waveletTransform.Daubechies_inverse_FWT_1d(removed)
    elif myCfg.method == "DAE":
        """ 3. AutoEncoder """
        x_input = np.concatenate((batch_data, np.zeros(shape=(2048-myCfg.window_length))), axis=0)
        y_input = np.concatenate((ori_data, np.zeros(shape=(2048-myCfg.window_length))), axis=0)
        denoised_data = denoisingAutoencoder.denoise(x_input, y_input)

    # calculate R with MSE
    # calcR = dist.MeanSquareError(denoised=denoised_data, noisy=batch_data)
    # kalman_filtering.set_R(calcR)

    # calculate R with Stdev
    # calcR = dist.StandardDeviation(denoised=denoised_data, noisy=batch_data)
    # kalman_filtering.set_R(calcR)

    # calculate R with Variance
    calcR = 0.0
    if myCfg.method == "MA" or myCfg.method == "WT" or myCfg.method == "DAE":
        calcR = dist.Variation(denoised=denoised_data, noisy=batch_data)
        # calcR = dist.StandardDeviation(denoised=denoised_data, noisy=batch_data)
    else:
        calcR = random.uniform(0.0, 90.0)
    kalman_filtering.set_R(calcR)
    R_sum += calcR

print("Euclidean Distance: {}".format(math.sqrt(total_dist)))
print("Measurement noise R Mean: {}".format(R_sum/(myCfg.total_windows-1)))
result_file.close()