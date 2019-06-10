# -*- coding: utf-8 -*-

data_root = "/Users/weidian/Documents/MyDiary/data/competition/huawei/"

# data for train
dir_data_train = data_root + './data_train'
dir_images_train = dir_data_train + '/images'
dir_contents_train = dir_data_train + '/contents'

# data for validation
dir_data_valid = data_root + './data_valid'
dir_images_valid = dir_data_valid + '/images'
dir_contents_valid = dir_data_valid + '/contents'
dir_results_valid = dir_data_valid + '/results'

model_detect_dir = data_root + './model_detect'
model_detect_name = 'model_detect'
model_detect_pb_file = model_detect_name + '.pb'

anchor_heights = [6, 12, 24, 36]

threshold = 0.5  #
