"""
Moving Average
"""

import MyConfiguration as myCfg

sensor_data = []

def MovingAverage():
    print(sensor_data)
    result = []
    sliding_window = int(myCfg.window_length / 4)
    sum = 0.0

    for i in range(len(sensor_data)):
        if i < sliding_window:
            sum += sensor_data[i]
            result.append(sum/sliding_window)
        else:
            sum += (sensor_data[i] - sensor_data[i-sliding_window])
            result.append(sum/sliding_window)
    return result