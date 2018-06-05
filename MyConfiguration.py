data_route = 'routes'
# data_route = 'D:/DKE/논문/파라미터 프리 칼만필터/DB연구저널/MA, WT 결과/stock_with_noise_MANY.csv'

# train data route
train_data_route = './training_data/dataset.json'
trained_model_route = "DaeTraining1/model/model_12(lr_01_e1000_adam)"

window_length = 32

total_windows = int(2049279/window_length)
# 512: 4002
# 1024: 2001
# 2048: 1000

original_row = 0
noisy_row = 1

# use method
method = "MA"

# AutoEncoder