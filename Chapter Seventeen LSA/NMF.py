# -*- coding:utf-8 -*- 
'''
作者:Mustang
日期:2021年08月17日
'''
import numpy as np
from sklearn.decomposition import TruncatedSVD
X = [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 2, 3], [0, 0, 0, 1], [1, 2, 2, 1]]
X = np.asarray(X);
print('X=',X)

# 非负矩阵分解算法
def inverse_transform (W, H):
    #重构
    return W.dot(H)

def loss(X, X_):
    # 计算重构误差
    return ((X - X_) * (X - X_)).sum()

# 算法 17.1


class MyNMF:
    def fit(self, X, k, t):
        m, n = X.shape

        W = np.random.rand(m, k)
        W = W/W.sum(axis=0)

        #W.sum(axis=0)表示列相加
        #W.sum(axis=1)表示行相加
        #W.sum()      表示所有元素相加

        H = np.random.rand(k, n)

        i = 1
        while i < t:

            W = W * X.dot(H.T) / W.dot(H).dot(H.T)

        #***验证mat1.dot(mat2)相当于dot(mat1,mat2)***
            # mat1 = np.array([[1,2,3],[4,5,6]])
            # mat2 = np.array([[1,2],[1,2],[1,2]])
            # np.dot(mat1,mat2)
            # array([[ 6, 12],
            # [15, 30]])
            # mat1.dot(mat2)
            # array([[ 6, 12],
            # [15, 30]])

            H = H * (W.T).dot(X) / (W.T).dot(W).dot(H)

            i += 1

        return W, H
model = MyNMF()
W, H = model.fit(X, 3, 200)

print('W=',W)
print('H=',H)


X_ = inverse_transform(W, H);      #重构

print('X_=',X_)

print('loss(X,X_)=',loss(X, X_))   #重构误差




