# -*- coding:utf-8 -*- 
'''
作者:Mustang
日期:2021年08月14日
'''
# import numpy as np
# np.dot(a, b, out=None)  #该函数的作用是获取两个元素a,b的乘积
# 数组的运算是元素级的，数据相乘的结果是各个元素的积组成的数组，而对于矩阵相乘，是点积。而Numpy库提供了用于矩阵乘法的dot函数
# dot()函数运算过程所示
"单个数的dot函数运算"
# import numpy as np
# np.dot(5,8)
# 40
"一维数组的dot函数运算"
# arr1 = np.array([2,3])
# arr2 = np.array([4,5])
# np.dot(arr1,arr2)
# 23
"二维数组dot()函数运算所示"
# 二维数组矩阵之间的dot()函数运算得到的乘积是矩阵乘积
# arr5 = np.array([[2,3],[4,5]])
# arr6 = np.array([[6,7],[8,9]])
# np.dot(arr5,arr6)
"二维数组与三维数组的dot()函数运算"
# arr7 = np.array([[2,3,4],[5,6,7]])
# arr8 = np.arange(9).reshape(3,3)
# np.dot(arr7, arr8)
# array([[33, 42, 51],
#       [60, 78, 96]])

##*************************************************************************##
# 为什么要用numpy
# python中提供了List容器，可当数组使用。但是列表中的元素可以是任何对象，因此列表中保存的是对象的指针，但是对于数组运算来说，这种结构显然不够高效
# Python也提供了array模块，但其支持一维数组，不支持多维数组，也没有各种运算函数。也不适合数值运算
# numpy弥补了这些不足

"数组创建"
## 常规创建方法
import numpy as np
# 一维数组
a = np.array([2,3,4])
b = np.array([2.,3.,4.])
# 二维数组
c = np.array([[1.,2.],[3.,4.]])
d = np.array([[1,2],[3,4]],dtype=complex)

print(a,a.dtype)
print(b,b.dtype)
print(c,c.dtype)
print(d,d.dtype)

"利用函数创建数组"
np.arange(0,7,1,dtype=float)    #arange 函数创建
array([0., 1., 2., 3., 4., 5., 6.])

np.ones((2,3,4),dtype=int)      #创建2页3行4列的数据
array([[[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]],
       [[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]])

np.zeros((2,3,4))  # 创建2页3行4列的数据
array([[[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]],
       [[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]]])

np.linspace(-1,2,5)  # 起点为-1，终点为2，取5个数
array([-1.  , -0.25,  0.5 ,  1.25,  2.  ])

np.random.randint(-9,3,(2,3))  #随机生成两行三列，大小为[-9,3)之间（左闭右开）的随机整数
array([[-8,  2, -2],
       [-6, -4, -1]])

"修改数据"
# 单个赋值/批量赋值/遍历赋值
#一维数组
a=np.arange(0,10,1)**2
array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

a[-1] = 100   #单个赋值
array([  0,   1,   4,   9,  16,  25,  36,  49,  64, 100])

a[1:4] = 100  #批量赋值
array([  0, 100, 100, 100,  16,  25,  36,  49,  64, 100])

b = [np.sqrt(np.abs(i)) for i in a]
print(b)
[0.0, 10.0, 10.0, 10.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

"数组输出"
# 从左到右，从上到下
# 一维数组输出为行，二维数组输出为矩阵，三维数组输出为矩阵列表

"基本运算"
# 数组运算&矩阵运算
# 元素级运算(一维数组)：加减/乘/平方/判断/三角函数等

a = np.arange(1,5,1)
b = np.array([0,1,2,3])
a - b           #对应相减
array([1, 1, 1, 1])
a * b           #对应相乘
array([ 0,  2,  6, 12])
a ** 2          #求平方
array([ 1,  4,  9, 16], dtype=int32)
np.sin(a) * 5   #三角函数
array([ 4.20735492,  4.54648713,  0.70560004, -3.78401248])
a > 3           #判断值
array([False, False, False,  True])
np.exp(a)       #指数
array([ 2.71828183,  7.3890561 , 20.08553692, 54.59815003])

"统计计算"
# 均值
a = np.random.randint(0,5,(2,3))          #生成一个2*3的矩阵，最大值不能超过5，最小值不能小于0
print(a.sum(),a.sum(0),a.sum(axis=0))     #分别对矩阵的行和列求和
9 [4 5 0] [4 5 0]
print(a.mean(),a.min(1),a.std(1))         # 输出均值，最小值，标准差
1.5 [0 0] [1.69967317 1.88561808]
np.median(a)                              # 输出中位数
0.5

"矩阵运算(二维数组)"
a = np.array([[1,2],[3,4]])
b = np.arange(6).reshape(2,-1)     #生成2行矩阵
print(a,'\n',b)
[[1 2]
 [3 4]]
 [[0 1 2]
 [3 4 5]]

"索引/切片/遍历"
# 数组的索引切片原理类似于list：索引从0开始，-1代表最后一个索引；左闭右开原则
# 一维数组
a = np.arange(0,10,1) ** 2
print(a[2],a[-1])
4 81
print(a[-3:-1])         #-3到-1，截取-1,输出-3至-2
[49 64]
print(a[0:-1])          #显然截去的是最后一个81
[ 0  1  4  9 16 25 36 49 64]

# 遍历输出
for i in a:
    print(i)

0
1
4
9
16
25
36
49
64
81

# 二维数组
c = np.arange(0,20,1).reshape(4,-1)
print('第二行：',c[1],'\n二到四列：\n',c[:,1:4],'\n二到四行的第三列',c[1:4,2],
      '\n第二行：',c[1,:])
第二行： [5 6 7 8 9]
二到四列：
 [[ 1  2  3]
 [ 6  7  8]
 [11 12 13]
 [16 17 18]]
二到四行的第三列 [ 7 12 17]
第二行： [5 6 7 8 9]

"形状操作"
a = 10 * np.random.random((3,4))    # 随机产生大小在0-1之间的3行4列的数
b = np.floor(a)                     # 截取整数部分

b.ravel()                           # 一维化操作(一行)
array([2., 6., 2., 3., 9., 2., 0., 6., 8., 3., 8., 0.])

b.shape=(6,-1)                      # 6行两列

b.transpose()                       # 转置
array([[2., 2., 9., 0., 8., 8.],
       [6., 3., 2., 6., 3., 0.]])

"删除"
# 一维数组中删除元素
a = np.arange(1,5,1)
print(a)
[1 2 3 4]
a = np.delete(a,0)  #删除a中的第一个元素
print(a)
[2 3 4]

# 二维数组中删除元素
# 注:在删除里axis = 0
b=np.arange(0,10,1).reshape(2,-1)
print(b)
[[0 1 2 3 4]
 [5 6 7 8 9]]

b = np.delete(b,1,axis=0)       # 删除b中第二行元素
print(b)
[[0 1 2 3 4]]

b=np.delete(b,2)                # 删除b中值为2的元素
print(b)
[0 1 3 4 5 6 7 8 9]

"axis 的本质"
# 简单来说，axis=0代表跨行，axis=1代表跨列
# 换言之，使用0值表示沿着每一列或行标签|索引值向下执行方法
#       使用1值表示沿着每一行或列标签模向执行对应的方法
import pandas as pd
df = pd.DataFrame([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], \
columns=["col1", "col2", "col3", "col4"])
df
   col1  col2  col3  col4
0     1     1     1     1
1     2     2     2     2
2     3     3     3     3

df.mean(axis=1)     #调用df.mean(axis=1),将得到按行计算的均值
0    1.0
1    2.0
2    3.0
dtype: float64

df.drop("col4",axis=1) #调用 df.drop((name, axis=1),删掉了一列，而不是一行：
   col1  col2  col3
0     1     1     1
1     2     2     2
2     3     3     3

"条件判断np.where()用法"
# 条件逻辑运算
# np.where (condition,arr1,arr2):condition 为真时，取arr1，反之取arr2
arr = np.random.randn(4,4)
arr_new = np.where(arr>0,1,-1)  #显示的是与0的判断，>0为1，<0为-1

# 多重条件判断
arr = np.random.randint(-100,200,size = (4,5))
arr
array([[-64,  48,  -5, 162, -41],
       [ 11, 152,  81, -53, 174],
       [125,   6, 110,  21, 168],
       [-96,  47, -28, -15, -63]])
arr_new = np.where(arr>=100,1, np.where(arr>=0,0,-1))
arr_new
array([[-1,  0, -1,  1, -1],
       [ 0,  1,  0, -1,  1],
       [ 1,  0,  1,  0,  1],
       [-1,
        0, -1, -1, -1]])

"python中arrange()和range()函数"

# 函数：range()
#函数说明： range(start, stop[, step]) -> range object，根据 start 与 stop 指定的范围以及 step 设定的步长，生成一个序列。
#参数含义：
#start: 计数从 start 开始。默认是从 0 开始。例如 range（5）等价于 range（0， 5）;
#end: 开始到 end 结束，不包括 end. 例如：range（0， 5） 是 [0, 1, 2, 3, 4] 没有 5
#scan：每次跳跃的间距，默认为 1。例如：range（0， 5） 等价于 range(0, 5, 1)
#range 多用作循环，range（0,10）返回一个 range 对象，如想返回一个 list，前面加上 list 转换
#函数返回的是一个 range object

>>> range(0,5) 			 	#生成一个range object,而不是[0,1,2,3,4]
range(0, 5)
>>> c = [i for i in range(0,5)] 	 #从0 开始到4，不包括5，默认的间隔为1
>>> c
[0, 1, 2, 3, 4]
>>> c = [i for i in range(0,5,3)] 	 #间隔设为3
>>> c
[0, 3]
range中的setp 不能使float,所有range不能生成小数。

# 函数：arrange()
#函数说明：arange([start,] stop[, step,], dtype=None) 根据 start 与 stop 指定的范围以及 step 设定的步长，生成一个 ndarray。 返回array 类型对象。

>>> np.arange(3)
    array([0, 1, 2])
>>> np.arange(3.0)
    array([ 0.,  1.,  2.])
>>> np.arange(3,7)
    array([3, 4, 5, 6])
>>> np.arange(3,7,2)
    array([3, 5])
range() 中的步长不能为小数，但是numpy.arange() 中的步长可以为小数
>>> arange(0,1,0.1)
array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])