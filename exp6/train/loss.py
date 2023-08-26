import math
import torch
from torch import Tensor, nn

_eps = 1e-6


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y, msk):
        loss = ((y_pred - y) ** 2) * msk
        return loss.sum() / (msk.sum() + _eps)


class MaskedPairwiseRankingLoss(nn.Module):
    ''' 带遮蔽的 Pairwise Ranking Loss'''

    def __init__(self, wd=0.1, exp=2):
        super().__init__()
        self.wd = wd
        self.exp = exp

    def forward(self, y_pred, y, msk):
        '''
        inputs
        -
        形状均为 (batch_size, time_step)
        '''

        y_pred = y_pred.transpose(0, 1)
        y = y.transpose(0, 1)
        msk = msk.transpose(0, 1)

        # 扩展后形状为 (time_step, batch_size, batch_size)
        exd_y_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(2)
        exd_y = y.unsqueeze(1) - y.unsqueeze(2)
        exd_msk = msk.unsqueeze(1) * msk.unsqueeze(2)
        calc_msk = (exd_y > 0) * exd_msk

        loss = torch.pow(
            torch.log(torch.sigmoid(exd_y_pred)) * calc_msk / math.log(0.5),
            self.exp
        ).sum()
        loss /= (calc_msk.sum() + _eps)

        if self.wd != 0:
            loss += ((y_pred ** 2).sum() / (msk.sum() + _eps)) * self.wd
        return loss


class VectorSimilarityLoss(nn.Module):
    ''' 计算向量的余弦相似度 '''

    def __init__(self):
        super().__init__()

    def forward(self, v_pred, v, msk):
        '''
        inputs
        -
        - v_pred 和 v 的形状均为 (batch_size, time_step, num_good_features)
        - msk 的形状为 (batch_size, time_step)
        '''

        dot_results = (v_pred * v).sum(dim=-1)
        len_v_pred = torch.sqrt((v_pred ** 2).sum(dim=-1))
        len_v = torch.sqrt((v ** 2).sum(dim=-1))
        cos_sim = dot_results / len_v_pred / len_v
        return ((1 - cos_sim) * msk).sum() / (msk.sum() + _eps)


class CombinedLoss(nn.Module):
    ''' 将三种 Loss 组合到一起 '''

    def __init__(self, alpha_mse, alpha_rank, alpha_cos, pred_wd=0.1, exp: int = 2):
        '''
        inputs
        -
        - alpha_mse: 平方差损失对应的权重
        - alpha_rank: Pairwise Ranking Loss 对应的权重
        - alpha_cos: 一个数或一个长度为 2 的元组。
          当传入一个长度为 2 的元组的时候将分别对应两天 good_features 的预测结果的权重
          当传入一个数的时候代表两天拥有相同的权重
        - pred_wd: Pairwise Ranking Loss 的 weight_decay 的取值
        - exp: Pairwise Ranking Loss 的 exp 参数
        '''
        super().__init__()
        self.mse = MaskedMSELoss()
        self.rank = MaskedPairwiseRankingLoss(pred_wd, exp)
        self.cos = VectorSimilarityLoss()
        self.alpha_mse = alpha_mse
        self.alpha_rank = alpha_rank
        if not isinstance(alpha_cos, (tuple, list)):
            self.alpha_cos = (alpha_cos, alpha_cos)
        else:
            self.alpha_cos = alpha_cos

    def forward(self, net_output, y, y_msk, X_day_1, msk_day_1, X_day_2, msk_day_2):
        '''
        inputs
        -
        - net_output: 神经网络的输出
        - y: 真实的 label, 形状为 (batch_size, time_step), 类型为 torch.float32
        - y_msk: 每个批次，时间步对应的数据是否存在, 形状为 (batch_size, time_step), 类型为 torch.bool
        - X_day_1: day + 1 时刻的 good_features, 形状为 (batch_size, time_step, num_good_features)，类型为 torch.float32。
          X_day_1[2, 3] 表示是在批次中第 3 个数据在第 5 个时间步的 good_features
        - msk_day_1: X_day_1 对应的每组数据是否真实存在，类型为 torch.bool
        - X_day_2: day + 2 时刻的 good_features, 形状为 (batch_size, time_step, num_good_features)，类型为 torch.float32。
        - msk_day_2: X_day_2 对应的每组数据是否真实存在，类型为 torch.bool

        returns
        -
        返回一个仅有 1 个元素, shape 为 [] 的 tensor, 表示该组数据的平均 loss
        '''
        y_pred, pred_day_1, pred_day_2 = net_output
        loss_label = self.alpha_mse * self.mse(y_pred, y, y_msk) + \
                     self.alpha_rank * self.rank(y_pred, y, y_msk)

        loss_vec = self.alpha_cos[0] * self.cos(pred_day_1, X_day_1, msk_day_1) + \
                   self.alpha_cos[1] * self.cos(pred_day_2, X_day_2, msk_day_2)

        loss = (loss_label + loss_vec) / (self.alpha_mse + self.alpha_rank + sum(self.alpha_cos))
        return loss

