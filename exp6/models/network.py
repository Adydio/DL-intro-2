import torch
from torch import nn

from .structure import AbstractTimestepModel, AbstractDimReduceModel, AbstractAnalyseModel, AbstractConnectModel


# 时间步模型

class GRUTimestepModel(AbstractTimestepModel):
    def __init__(self, num_inputs, num_hiddens, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens
        self.rnn_layer = nn.GRU(num_inputs, num_hiddens, num_layers)
        self.W_o1 = nn.Linear(num_hiddens, num_inputs)
        self.W_o2 = nn.Linear(num_hiddens, num_inputs)

    def begin_state(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.num_hiddens, device=device)

    def forward(self, X, *args):
        # 交换时间和批次轴
        X = X.transpose(0, 1)
        H = self.begin_state(X.shape[1], device=X.device)

        pred1, pred2 = [], []
        for t in range(len(X)):
            X_day_1, H = self.rnn_layer(X[t: t + 1], H)
            pred_day1 = self.W_o1(X_day_1)
            X_day_2, _ = self.rnn_layer(pred_day1, H)
            pred_day2 = self.W_o2(X_day_2)

            pred1.append(pred_day1)
            pred2.append(pred_day2)

        pred1 = torch.concat(pred1, dim=0).transpose(0, 1)
        pred2 = torch.concat(pred2, dim=0).transpose(0, 1)
        return pred1, pred2


# 降维模型

class LinearFeedbackNetwork(AbstractDimReduceModel):
    def __init__(self, size_list, **kwargs):
        super().__init__(*kwargs)
        last, size_list = size_list[0], size_list[1:]
        nets = []
        for next_ in size_list:
            nets.append(nn.Linear(last, next_))
            nets.append(nn.ReLU())
            last = next_
        nets = nets[: -1]
        self.seq = nn.Sequential(*nets)

    def forward(self, X, *args):
        return self.seq(X)


# 分析模型
'''
这里可能能再用点 Attention
'''


class NaiveSubAnalyseModel(AbstractAnalyseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X_day_0, X_day_1, X_day_2, *args):
        return X_day_2 - X_day_0


# 连接模型

class LinearConnectModel(AbstractConnectModel):
    def __init__(self, num_hiddens, size_list, **kwargs):
        super().__init__()
        num_inputs = sum(size_list)
        self.lin1 = nn.Linear(num_inputs, num_hiddens)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(num_hiddens, 1)

    def forward(self, *args):
        X = torch.concat(args, dim=-1)
        return self.lin2(self.relu(self.lin1(X)))