"""Print all variables saved in model checkpoint.
"""
import os
from tensorflow.python import pywrap_tensorflow

model_dir = '/home/liuhy/Data/models/facenet/20180402-114759'
checkpoint_path = os.path.join(model_dir, "model-20180402-114759.ckpt-275")

# checkpoint_path = './saved_model/model-10'

# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

# var_list = []
# Print tensor name and values
# for key in var_to_shape_map:
    # print("tensor_name: ", key)
    # var_list.append(key)

with open('../data/structure.txt', 'w') as f:
    for key in var_to_shape_map:
        f.writelines(key + '\n')