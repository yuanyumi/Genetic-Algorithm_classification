# 遗传算法或梯度下降法的分类问题

## 问题描述
根据60个特征（0~1）分类为岩石（R）和金属（M）

网址：http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)

## 运行方式
python train.py


## Tips
可设置参数： 

if  use_ga = True :使用遗传算法进行权重优化

else:使用梯度下降法进行权重优化

可用于其他数据集，适用多分类任务，需要修改数据预处理部分（独热编码）以及神经网络输出层（output_size）