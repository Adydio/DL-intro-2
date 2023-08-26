import torch
import random
from .cleanlib import normalize
from .dataset import to_table, extra_label_from_tuple
from .base import AbstractDataset


class Dataset(AbstractDataset):

    def __init__(self, df, has_label=True, **kwargs):
        df = normalize(df)
        self.data = to_table(df)
        self.splitted = False
        self.has_label = has_label
        self.num_features = self.data[2].shape[-1]
        self.label = extra_label_from_tuple(self.data)

    def split_train_valid(self, ratio, **kwargs):
        self.data = split_train_test_by_time(*self.data, ratio=ratio)
        self.label = tuple(extra_label_from_tuple(x) for x in self.data)
        self.splitted = True

    def get_dataloader(self, batch_size, window_size, device, **kwargs):
        kwargs['batch_size'] = batch_size
        kwargs['window_size'] = window_size
        kwargs['device'] = device
        if not self.splitted:
            return DataLoader(self.data, **kwargs)
        train, valid = self.data
        train_iter = DataLoader(train, **kwargs)
        test_iter = DataLoader(valid, **kwargs)
        return train_iter, test_iter

    def get_label(self):
        return self.label


# 这个目前是照搬的，还没改
def split_train_test_by_time(*data, ratio):
    '''
    直接根据时间维度进行划分训练集和测试集
    '''

    cnt_train = int(data[0].shape[0] * (1 - ratio))
    train = tuple(x[: cnt_train] for x in data)
    test = tuple(x[cnt_train:] for x in data)
    return train, test


# 这个也是照搬的，还没改，主要是要写那个get_data_iter
class DataLoader:
    def __init__(self, data, batch_size, device, **kwargs):
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.params = kwargs
        self.iter_fn = get_timestep_data_iter

    def __iter__(self):
        params = {'batch_size': self.batch_size, 'device': self.device}
        if isinstance(self.data, tuple):
            return self.iter_fn(*self.data, **params, **self.params)
        return self.iter_fn(self.data, **params, **self.params)


# 这个应该是咱们要写的最关键部分
def get_timestep_data_iter(*data, batch_size, time_step, device, **kwargs):
    '''
    关于迭代器生成器的一些要求：
    * 参数列表形如 get_iter(*data, batch_size, time_step, device, **kwargs)
      其中 *data 处会传入数据集构成的元组，格式和 Dataset 构建时转成 torch 张量时的存储格式相同
      相当于data就是to_table这个函数的输出，加星号就是解包（个人理解）
    * 生成的迭代器的格式需要对接 models.structure.BasicExcutor.forward 的输入以及 train.loss.CombinedLoss 的输入
      具体来说在 train 中希望通过下面的代码计算由迭代器生成的小批次数据 X 中损失
      X, y = X[: -1], X[-1]
      l = loss(net(*X), *y)
    * 每个批次的数据需要在时间步上对齐，例如 features[1, t, :] 和 features[3, t, :] 需要在同一天
    * 如果没有足够的数据，返回的小批次的数据的 batch_size 和 time_step 可以比设定值小
    '''
    return 0
