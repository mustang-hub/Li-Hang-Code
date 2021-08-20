# -*- coding:utf-8 -*- 
'''
作者:Mustang
日期:2021年08月6日
'''
# singular-value decomposition
# SVD 可以通过调用svd()函数计算，返回U、Sigma和V^T元素
from numpy import array
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print('A=',A)
# SVD
U, s, VT = svd(A)
print('U=',U)
print('s=',s)
print('VT=',VT)
