# regression-model-as-a-surrrogate-model
Use regression model as a surrogate to tackle a kind of problems named parameter estimation

## 数据集

数值仿真采样10000组输入空间的样本，还包括每个参数下的压力、体积和应变数据。

输入空间4维，输出特征是通过对压力体积和压力应变拟合得到的拟合参数。

### Fisrt run

1. clone respository
2. pip install -r requirements
3. 把datasets.xlsx文件放在根目录
4. 在 jupyter notebook 环境下运行 regression_surrogate.ipynb
