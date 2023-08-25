import torch
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