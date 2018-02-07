import math

def MeanSquareError(denoised, noisy):
    sum = 0.0
    for idx in range(len(denoised)):
        sum += math.pow((denoised[idx] - noisy[idx]), 2)

    return math.sqrt(sum)


def _STDEV():

    return 0