import json
import random
import copy
import numpy as np

class NoiseMask:
    def __init__(self, input_url):
        self.input_url = input_url
        # './training_data/dataset.json'
        self.original_list = []
        self.o_training_list = []
        self.o_validation_list = []
        self.o_test_list = []
        self.data_stdev = 0.0
        self.isFirst = True

    # 표준편차*표준편차(=분산)의 n% 사용
    def _generateNoise(self, data_dicts, data_stdev):
        noise_list = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
        for d in data_dicts:
            series = d['time_series'].copy()
            noise_idx = random.randint(0, len(noise_list)-1)
            for i, val in enumerate(series):
                rand = random.gauss(0.0, data_stdev*data_stdev * noise_list[noise_idx])
                val += rand
                series[i] = val
            d['time_series'] = series

        return data_dicts

    def _zeroPadding(self, data_dicts):
        result_list = []
        for d in data_dicts:
            now_lenght = d['length']
            time_series = np.array(d['time_series'])
            result_list.append(np.concatenate((time_series, np.zeros(2048-now_lenght))))

        return np.array(result_list)

    def _readJson(self):
        print("I'm Json Reader!")
        with open(self.input_url) as json_file:
            json_data = json.load(json_file)
            # print(json_data['data'])

            original_json_data = json_data['data']

            # get standard dev
            self.data_stdev = json_data['stdev']
            lens = original_json_data[1]['length']
            # print(self.data_stdev)
            # print(lens)


            original_list = []
            for l in original_json_data:
                original_list.append(l)
        self.original_list = original_list


    def _divideData(self):
        # first shuffle (time series length 다르게!)
        random.shuffle(self.original_list)
        self.o_training_list = copy.deepcopy(self.original_list[:int(12000*0.6)])
        self.o_validation_list = copy.deepcopy(self.original_list[int(12000*0.6):int(12000*0.8)])
        self.o_test_list = copy.deepcopy(self.original_list[int(12000*0.8):])


    def createTrainingData(self):
        if self.isFirst:
            self.isFirst = False
            self._readJson()
            self._divideData()
        original_list = copy.deepcopy(self.o_training_list)
        random.shuffle(original_list)

        noisy_list = self._generateNoise(copy.deepcopy(original_list), self.data_stdev)
        # print(len(noisy_list))

        original_time_series = self._zeroPadding(original_list)
        noisy_time_series = self._zeroPadding(noisy_list)

        # print("Origin shape: {}, Noisy shape: {}".format(original_time_series.shape, noisy_time_series.shape))
        return noisy_time_series, original_time_series


    def createValidationData(self):
        validation_list = copy.deepcopy(self.o_validation_list)

        noisy_list = self._generateNoise(copy.deepcopy(validation_list), self.data_stdev)
        validation_time_series = self._zeroPadding(validation_list)
        noisy_time_series = self._zeroPadding(noisy_list)

        return noisy_time_series, validation_time_series

# noiseMask = NoiseMask('./training_data/dataset.json')
# x, y = noiseMask.createTrainingData()