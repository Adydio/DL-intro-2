import pandas as pd
import json


def read_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


# 不需要传入df
def feature_selector(num):
    if num == 0:
        return []
    m = num // 2 + 1
    day1 = read_json('./exp6/day+1.json')
    day2 = read_json('./exp6/day+2.json')
    corr_matrix = pd.DataFrame(read_json('./exp6/corr_matrix.json'))
    # 合并这两天的特征
    combined_features = {k: abs(day1.get(k, 0)) + abs(day2.get(k, 0)) for k in set(day1) | set(day2)}
    # 按绝对值大小排序并选出前m个特征
    sorted_features = sorted(combined_features.items(), key=lambda x: abs(x[1]), reverse=True)
    m_features = [x[0] for x in sorted_features[:m]]
    if m == 1:
        return m_features
    all_correlations = []
    for feature in m_features:
        # 找出相关系数的绝对值，并除去与自己的相关系数
        correlations = corr_matrix[feature].drop(feature).abs()
        all_correlations.extend(correlations.items())
    # 排序
    all_correlations = sorted(all_correlations, key=lambda x: x[1], reverse=True)
    related_features = []
    for feature, correlation in all_correlations:
        # 如果特征不在m_features和related_features中，则添加到related_features中
        if feature not in m_features and feature not in related_features:
            related_features.append(feature)
        # 一旦related_features的长度达到num-m，跳出循环
        if len(related_features) == num-m:
            break

    return m_features + related_features