import torch
import random
from .cleanlib import normalize
from .dataset import to_table, extra_label_from_tuple
from .base import AbstractDataset


class Dataset(AbstractDataset):

    def __init__(self, df, has_label=True, **kwargs):
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


def get_timestep_data_iter(*data, batch_size, time_step, device, **kwargs):
    has_label = len(data) == 6
    if has_label:
        T, I, X, X_selected, msk, label = data
    else:
        T, I, X, X_selected, msk = data

    offset = random.randint(0, time_step - 1)
    total_timesteps = T.shape[0]
    total_stocks = T.shape[1]

    for t in range(offset, total_timesteps - time_step - 2, time_step):
        for start_stock in range(0, total_stocks, batch_size):
            end_stock = start_stock + batch_size

            batch_T = T[t:t+time_step, start_stock:min(end_stock, total_stocks)]
            batch_I = I[t:t+time_step, start_stock:min(end_stock, total_stocks)]
            batch_X = X[t:t+time_step, start_stock:min(end_stock, total_stocks)]
            batch_X_selected = X_selected[t:t+time_step, start_stock:min(end_stock, total_stocks)]
            batch_msk = msk[t:t+time_step, start_stock:min(end_stock, total_stocks)]

            batch_T = batch_T.permute(1, 0) if len(batch_T.shape) == 2 else batch_T.permute(1, 0, 2)
            batch_I = batch_I.permute(1, 0) if len(batch_I.shape) == 2 else batch_I.permute(1, 0, 2)
            batch_X = batch_X.permute(1, 0) if len(batch_X.shape) == 2 else batch_X.permute(1, 0, 2)
            batch_X_selected = batch_X_selected.permute(1, 0) if len(batch_X_selected.shape) == 2 else batch_X_selected.permute(1, 0, 2)
            batch_msk = batch_msk.permute(1, 0) if len(batch_msk.shape) == 2 else batch_msk.permute(1, 0, 2)

            if has_label:
                batch_label = label[t:t+time_step, start_stock:min(end_stock, total_stocks)]
                batch_label = batch_label.permute(1, 0) if len(batch_label.shape) == 2 else batch_label.permute(1, 0, 2)

                y = batch_label
                y_msk = batch_msk
                X_day_1 = X_selected[t+1:t+time_step+1, start_stock:min(end_stock, total_stocks)]
                msk_day_1 = msk[t+1:t+time_step+1, start_stock:min(end_stock, total_stocks)]
                X_day_2 = X_selected[t+2:t+time_step+2, start_stock:min(end_stock, total_stocks)]
                msk_day_2 = msk[t+2:t+time_step+2, start_stock:min(end_stock, total_stocks)]

                X_day_1 = X_day_1.permute(1, 0) if len(X_day_1.shape) == 2 else X_day_1.permute(1, 0, 2)
                msk_day_1 = msk_day_1.permute(1, 0) if len(msk_day_1.shape) == 2 else msk_day_1.permute(1, 0, 2)
                X_day_2 = X_day_2.permute(1, 0) if len(X_day_2.shape) == 2 else X_day_2.permute(1, 0, 2)
                msk_day_2 = msk_day_2.permute(1, 0) if len(msk_day_2.shape) == 2 else msk_day_2.permute(1, 0, 2)

                y_tuple = (y, y_msk, X_day_1, msk_day_1, X_day_2, msk_day_2)
                yield batch_T, batch_I, batch_X, batch_X_selected, batch_msk, y_tuple
            else:
                yield batch_T, batch_I, batch_X, batch_X_selected, batch_msk