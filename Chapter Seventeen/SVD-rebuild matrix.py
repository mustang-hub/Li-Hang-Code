# -*- coding:utf-8 -*- 
'''
作者:Mustang
日期:2021年08月8日
'''
# svd() 返回的U、s和V元素不能直接相乘，s向量必须使用diag()函数转化成对角矩阵。在默认情况下，这个函数将创建一个相对于原来矩阵的m*m
# 的方形矩阵。但存在问题，因为该矩阵得尺寸并不符合矩阵乘法得规则。也就是行列不匹配
# 在创建了方形的Sigma对角矩阵之后，各个矩阵的大小与我们分解的原始n*m矩阵是相关的：
# U(m*n) . Sigma (n * n) .V^T (n * n)
# 而实际上，通过创建一个全是0值的m*n的新Sigma矩阵，并且通过diag()计算得到的方形对角矩阵来填充矩阵的前n*n部分

# Reconstruct SVD
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print('A=',A)
# Singular-value decomposition
U, s, VT = svd(A)
# create m * n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))    #返回给定形状和类型的新数组，用0填充
# populates Sigma with n*n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
#  reconstruct matrix
B = U.dot(Sigma.dot(VT))
print('B=',B)
# 运行这个示例，首先会显示原始矩阵，然后会显示根据 SVD 元素重建的矩阵。

# 如果sigma对角矩阵是m=n的话，Sigma = diag(s)
