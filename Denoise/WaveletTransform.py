import math
import numpy as np
import copy

#/*---------------------------------------------------------------------------*/
#/*
#Calculate the Daubechies forward fast wavelet transform in 1-dimension.
#*/

#/*------------------------------zzzzzzzzzzzzzzzzzzzzzz---------------------------------------------*/

class WaveletTransform:
    def __init__(self, n):
        self.n = n
        self.DataSet_dwt_daubechies = np.zeros(shape=(n))
        self.Inverse_dwt_daubechies = np.zeros(shape=(n))



    def Daubechies_forward_FWT_1d (self, input_list):

        k = int(math.log(self.n, 2))

        for m in range(self.n):
            self.DataSet_dwt_daubechies[m] = input_list[m]

        for m in range(k-1, -1, -1):
            self.daubechies_forward_pass_1d(m+1)

        return self.DataSet_dwt_daubechies

#####################################################################################

    def daubechies_forward_pass_1d (self, n):
        H = [0.683013, 1.18301, 0.316987, -0.183013 ]

        npts = int(math.pow(2, n))
        half = int(npts/2)

        a = np.zeros(shape=(half))
        c = np.zeros(shape=(half))

        for i in range(half):
            a[i] = (H[0]*self.DataSet_dwt_daubechies[int(2*i)%npts] + H[1]*self.DataSet_dwt_daubechies[int(2*i+1)%npts] + H[2]*self.DataSet_dwt_daubechies[int(2*i+2)%npts] + H[3]*self.DataSet_dwt_daubechies[int(2*i+3)%npts]) / 2.0
            c[i] = (H[3]*self.DataSet_dwt_daubechies[int(2*i)%npts] - H[2]*self.DataSet_dwt_daubechies[int(2*i+1)%npts] + H[1]*self.DataSet_dwt_daubechies[int(2*i+2)%npts] - H[0]*self.DataSet_dwt_daubechies[int(2*i+3)%npts]) / 2.0

        for i in range(half):
            self.DataSet_dwt_daubechies[i] = a[i]
            self.DataSet_dwt_daubechies[i + half] = c[i]

        a = []
        c = []

#/*---------------------------------------------------------------------------*/
#/*
#Calculate the Daubechies inverse fast wavelet transform in 1-dimension.
#*/

    def Daubechies_inverse_FWT_1d (self, input_list):

        k = int(math.log(self.n, 2))

        for m in range(self.n):
            self.Inverse_dwt_daubechies[m] = input_list[m]

        for m in range(k):
            self.daubechies_inverse_pass_1d(m+1)
            # print(self.Inverse_dwt_daubechies)

        return self.Inverse_dwt_daubechies

########################################################################################################

    def daubechies_inverse_pass_1d (self, n):
        H = [0.683013, 1.18301, 0.316987, -0.183013]
        npts = int(math.pow(2, n))
        half = int(npts/2)
        c = self.Inverse_dwt_daubechies[half:half+half]

        temp = np.zeros(shape=(npts))

        for i in range(half):
            temp[int(2*i)] = H[2]*self.Inverse_dwt_daubechies[(i-1+half)%half] + H[1]*c[(i-1+half)%half] + H[0]*self.Inverse_dwt_daubechies[i] + H[3]*c[i]
            temp[int(2*i+1)] = H[3]*self.Inverse_dwt_daubechies[(i-1+half)%half] - H[0]*c[(i-1+half)%half] + H[1]*self.Inverse_dwt_daubechies[i] - H[2]*c[i]

        for i in range(npts):
            self.Inverse_dwt_daubechies[i] = temp[i]

        temp = []
        c = []