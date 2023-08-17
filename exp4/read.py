import json
import numpy as np


# 返回的key是int型，value是embedding的list，类型为float，长度取决于文件
def read_dict(path):
    with open(path, 'r') as file:
        data = json.load(file)
    normalized_data = {}
    for key, value in data.items():
        normalized_vector = np.array(value) / np.linalg.norm(value, 2)  # L2 normalization
        normalized_data[int(key)] = normalized_vector.tolist()  # Convert numpy array back to list
    return normalized_data
