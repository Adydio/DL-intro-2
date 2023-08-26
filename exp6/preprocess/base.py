
from abc import abstractmethod


class AbstractDataset :

  def __init__(self, df, **kwargs) :
    '''
    将给定的 DataFrame 数据根据给定参数进行填充、标准化，然后将其转化为 torch 张量存储在内部
    '''
    pass
  
  @abstractmethod
  def split_train_valid(self, ratio, **kwargs) :
    '''
    将内部的数据集划分为训练集和验证集
    
    ratio: 划分出的验证集所占的比例
    '''
    pass
  
  @abstractmethod
  def get_dataloader(self, **kwargs) :
    '''
    根据内部存储的 torch 张量和给定的参数生成所需要的 DataLoader
    
    如果已经划分过训练集和验证集，那么需要返回一对 DataLoader
    '''
    pass
  
  @abstractmethod
  def get_label(self) : 
    '''
    从数据集中单独抽取出 label，形状是一个 dict: (time_id, stock_id) -> 对应的 label
    
    可以参考一下 preprocess.dataset.extra_label_from_tuple
    
    这一部分主要是用于提取测试集或者验证集的 label 用于 model.score()
    '''
    pass