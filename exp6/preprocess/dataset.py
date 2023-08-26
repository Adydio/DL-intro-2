import torch, random
from ..select import featureSelector


def to_table(df, num):
    '''
    输入：dataframe和所需的特征的数目
    返回表状数据集
    第一维为时间步，平移后使得最小值为 0
    第二维为离散化后的 stock
    第三维为数据维
    --------
    如果不含 label，那么将返回一个元组表示 (T, I, X, X_selected, msk)，分别表示 time_id, stock_id, 特征, 选定的特征以及该状态是否存在
    如果 df 中包含 label，那么将额外返回 label
    对于不存在的数据将以全零填充
    '''
    selected_features_init = featureSelector.feature_selector(num)
    selected_features = [int(i) for i in selected_features_init]
    sorted_features = sorted(selected_features) # 统一升序排列这些选定特征
    has_label = True
    if not 'label' in df.columns:
        has_label = False
    ltime, rtime = min(x for x, y in df.index), max(x for x, y in df.index)

    # 离散化 stock_id
    vis, index_to_id = set(), []
    for x, y in df.index:
        if not y in vis:
            index_to_id.append(y)
            vis.add(y)
    index_to_id = sorted(index_to_id)
    id_to_index = {x: i for i, x in enumerate(index_to_id)}

    table = torch.zeros(int(rtime - ltime + 1), len(id_to_index), len(df.columns) + 3 + num, dtype=torch.float32)
    for _i, x in df.iterrows():
        t, i = _i
        tt, ti = t - ltime, id_to_index[i]
        table[tt, ti, 2: 2+len(df.columns)] = torch.tensor(x.to_numpy(), dtype=torch.float32)
        table[tt, ti, 0], table[tt, ti, 1] = t, i
        table[tt, ti, -1] = 1
        table[tt, ti, 2+len(df.columns): -1] = torch.tensor(x.iloc[sorted_features], dtype=torch.float32)

    X = table[:, :, 2: -1]
    T, I = table[:, :, 0], table[:, :, 1]
    msk = table[:, :, -1]
    T, I = T.to(dtype=torch.int32), I.to(dtype=torch.int32)
    msk = msk.to(dtype=torch.bool)

    if has_label:
        X, label = X[:, :, : -1], X[:, :, -1]
        len_features = len(df.columns) - 1
    else:
        len_features = len(df.columns)

    X, X_selected = X[:, :, : len_features], X[:, :, len_features:]
    # 填充 T
    for i in range(len(T)):
        T[i] = max(T[i])
    # 填充 I
    for i in range(I.shape[1]):
        I[:, i] = max(I[:, i])

    result = T, I, X, X_selected, msk

    if has_label:
        result += (label,)
    return result


def split_train_valid(*data, ratio, random_start=False, index=None):
    '''
    将 data 划分为训练集和验证集，其中 ratio 是验证集的大小占比。data 其中每一项必须能支持切片以及用 + 连接
    为了方便以及和测试集类似，截取一段时间步中的股票信息作为验证集
    index: 当 index 非 None 的时候将返回 valid 对应的那一段的 index，这一段 index 是一维的
    '''
    n_sample = len(data[0])
    n_valid = int(n_sample * ratio)
    if random_start:
        start = random.randint(0, n_sample - n_valid)
    else:
        start = n_sample - n_valid
    valid = tuple(x[start: start + n_valid] for x in data)
    train = tuple()
    for x in data:
        if start + n_valid < n_sample:
            train += (x[: start] + x[start + n_valid:],)
        else:
            train += (x[: start],)
        if index is not None:
            L = sum(len(x[i]) for i in range(0, start))
            M = sum(len(x[i]) for i in range(start, start + n_valid))
            index_slice = index[L: L + M]
    result = (train, valid)
    if not index is None:
        result += (index_slice,)
    return result


# 获取 label

def _torch_type_switch(x, result_type):
    '''
    将 torch 张量转化为其他类型

    result_type:
      - 'float': 浮点数
      - 'np': numpy 数组
      - 'torch': 在梯度计算图之外的 torch 张量
    '''
    x = x.detach().cpu()
    if result_type == 'np':
        x = x.numpy()
    if result_type == 'float':
        x = float(x)
    return x


def extra_label_from_tuple(data, result_type='float'):
    '''
    data: data 的形状：(T, I, ..., msk, label)

    result_type: 默认为 float，可选 'float', 'np', 'torch'
    '''
    T, I, msk, label = *data[0: 2], *data[-2:]
    T, I, label = T[msk], I[msk], label[msk]
    result = {(int(t), int(i)): _torch_type_switch(y, result_type) for t, i, y in zip(T, I, label)}
    return result