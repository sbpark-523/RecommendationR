import math
import numpy as np

class DistanceCalculator:
    def __init__(self, write):
        self.write = write

    def MeanSquareError(self, denoised, noisy):
        sum = 0.0
        for idx in range(len(denoised)):
            sum += math.pow((denoised[idx] - noisy[idx]), 2)
            self.write.writelines('{}\n'.format(str(noisy[idx] - denoised[idx])))

        return math.sqrt(sum)


    def StandardDeviation(self, denoised, noisy):
        # 각 노이즈 차이 구함
        noise_list = []
        noise_mean = 0.0
        for i, denoisy in enumerate(denoised):
            point_dist = noisy[i] - denoisy
            noise_list.append(point_dist)
            # 노이즈 값의 평균 구함
            noise_mean += point_dist / len(denoised)

        # 표준편차 계산
        noise_stdev = np.std(noise_list)
        print("Mean: {}, Stdev: {}".format(noise_mean, noise_stdev))
        return noise_stdev

    def Variation(self, denoised, noisy):
        # calculate the "denoised - noisy" -->
        noise_list = []
        noise_mean = 0.0
        for i, denoisy in enumerate(denoised):
            point_dist = noisy[i] - denoisy
            noise_list.append(point_dist)
        noise_var = np.var(noise_list)
        return noise_var
