import numpy as np

import MyConfiguration as myCfg
import Denoise.MovingAverage as ma
import Denoise.WaveletTransform as wt
import DistanceCalculator as dist
import KalmanFilter

data = np.loadtxt(myCfg.data_route, delimiter=',', dtype=np.float32)

print("Load data from:", myCfg.data_route, data.shape)
# print(data)

result_file = open('./myresult.csv', 'w')

reshaped_data = np.transpose(data)
windows = int(len(reshaped_data[0])/myCfg.window_length)
kalman_filtering = KalmanFilter.KalmanFilter(result_file)
result = []

quarter = int(myCfg.window_length/4)

for x in range(8):
    batch_data = reshaped_data[myCfg.noisy_row, int(x * myCfg.window_length): int((x+1) * myCfg.window_length)]

    if x > 0:
        # print('Noisy: {}'.format(batch_data))
        result = kalman_filtering.filter(batch_data)
        print('Filtered: {}'.format(result))

    """ 1. Moving Average """
    # ma.sensor_data = batch_data
    # denoised_data = ma.MovingAverage()

    """ 2. Wavelet transform """
    waveletTransform = wt.WaveletTransform(myCfg.window_length)
    transformed = waveletTransform.Daubechies_forward_FWT_1d(batch_data)
    removed = np.concatenate((transformed[:quarter], np.zeros(shape=(len(transformed)-quarter))), axis=0)
    denoised_data = waveletTransform.Daubechies_inverse_FWT_1d(removed)


    # 3. AutoEncoder

    # calculate R with MSE
    calcR = dist.MeanSquareError(denoised=denoised_data, noisy=batch_data)
    kalman_filtering.set_R(calcR)

result_file.close()