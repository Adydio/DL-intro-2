import pandas as pd
import numpy as np
import json


def read_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


# 无需传入df，因为已经把相关性信息写入文件了
def feature_selector(num):
    m = num // 2 + 1
    day1 = read_json('./exp6/day+1.json')
    day2 = read_json('./exp6/day+2.json')
    corr_matrix = pd.DataFrame(read_json('./exp6/corr_matrix.json'))

    combined_features = {k: abs(day1.get(k, 0)) + abs(day2.get(k, 0)) for k in set(day1) | set(day2)}
    sorted_features = sorted(combined_features.items(), key=lambda x: abs(x[1]), reverse=True)
    m_features = [x[0] for x in sorted_features[:m]]

    all_correlations = []

    for feature in m_features:
        # 找出相关系数的绝对值，并除去与自己的相关系数
        correlations = corr_matrix[feature].drop(feature).abs()
        all_correlations.extend(correlations.items())
    # 排序并取前n-m个
    all_correlations = sorted(all_correlations, key=lambda x: x[1], reverse=True)
    related_features = [x[0] for x in all_correlations if x[0] not in m_features][:num - m]

    return m_features + related_features
