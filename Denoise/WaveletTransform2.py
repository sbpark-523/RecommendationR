import pywt

for i in range(10-1,-1, -1):
    print(i)
print()
for i in range(1,10, 1):
    print(i)


#
# sensor = [13.3, 8.4, 10.4, 16.1, 7.2, 15.9, 9.1, 16.4]
# print(sensor)
# mywavelet = pywt.Wavelet(name='db4')
# mywavelet.dec_len = 4
# mywavelet.rec_len = 4
# cA, cD = pywt.dwt(data=sensor, wavelet=mywavelet)
# # print(pywt.Wavelet(name='db1').dec_len)
#
# print(cA)
# print(cD)
#
# for i in range(len(cA)):
#     if i >= 2:
#         cA[i] = 0.0000001
#         # cD[i] = 0
#
# # print(cA)
# # print(cD)
#
# print(cA + cD)
#
# print(pywt.idwt(cA=cA, cD=None, wavelet=mywavelet))
