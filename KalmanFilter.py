# initialize

####### 이부분 전면 수정!!! class로..? 전역 변수가 안돼..ㅠ

class KalmanFilter:
    def __init__(self, writer):
        self.K = 0
        self.P = 1000
        self.Q = 0.01
        self.R = 0.5
        self.x = 0
        self.H = 1
        self.writer = writer

    def set_R(self, R):
        self.R = 0.5

    def filter(self, sensor):
        print('Kalman Filtering R: ', self.R)
        result = []
        for z in sensor:
            # predict
            x_next = self.x
            P_next = self.P + self.Q

            # correct
            self.K = P_next * self.H / (self.H * self.H * P_next + self.R)
            self.x = x_next + self.K * (z - self.H * x_next)
            self.P = (1 - self.K * self.H) * P_next
            result.append(self.x)

            self.writer.writelines('{}\n'.format(str(self.x)))

        return result
