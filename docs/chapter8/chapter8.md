# NumPy 介绍

NumPy是Python中科学计算的基础包。它是一个Python库，提供多维数组对象，各种派生对象（如掩码数组和矩阵），以及用于数组快速操作的各种API，有包括数学、逻辑、形状操作、排序、选择、输入输出、离散傅立叶变换、基本线性代数，基本统计运算和随机模拟等等。

NumPy包的核心是 *ndarray* 对象。它封装了python原生的同数据类型的 *n* 维数组，为了保证其性能优良，其中有许多操作都是代码在本地进行编译后执行的。

## 生成NumPy数组

NumPy是Python的外部库，不在标准库中。因此，若要使用它，需要先导入NumPy。

```python
import numpy as np
```

导入NumPy后，可通过np.+ `Tab` 键，查看可使用的函数，如果对其中一些函数的使用不很清楚，还可以在对应函数+`?`，再运行，就可很方便的看到如何使用函数的帮助信息。
 np.然后按`Tab`键，将出现如下界面：

<div align="center"><img src="chapter8/res/chapter8-1.png"></div>

### 利用random模块生成数组

<div align="center"><img src="chapter8/res/chapter8-2.png"></div>

### 创建特定形状的多维数组

<div align="center"><img src="chapter8/res/chapter8-3.png"></div>

### 利用 arange、linspace 函数生成数组

arange 是 numpy 模块中的函数，其格式为:

```python
arange([start,] stop[,step,], dtype=None)
```

其中start 与 stop 指定范围，step 设定步长，生成一个 ndarray，start 默认为 0，步长 step 可为小数。Python有个内置函数range功能与此类似。

```python
import numpy as np 
print(np.arange(10)) 
# [0 1 2 3 4 5 6 7 8 9] 
print(np.arange(0, 10)) 
# [0 1 2 3 4 5 6 7 8 9] 
print(np.arange(1, 4, 0.5)) 
# [1.  1.5 2.  2.5 3.  3.5] 
print(np.arange(9, -1, -1)) 
# [9 8 7 6 5 4 3 2 1 0]
```

linspace 也是 numpy 模块中常用的函数，其格式为:

```python
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```

它可以根据输入的指定数据范围以及等份数量，自动生成一个线性等分向量，其中endpoint (包含终点)默认为 True，等分数量num默认为 50。如果将retstep设置为 True，则会返回一个带步长的 ndarray。

```python
import numpy as np
 
print(np.linspace(0, 1, 10))
#[0.         0.11111111 0.22222222 0.33333333 0.44444444 0.55555556
# 0.66666667 0.77777778 0.88888889 1.        ]
```

值得一提的，这里并没有像我们预期的那样，生成 0.1, 0.2, ... 1.0 这样步长为0.1的 ndarray，这是因为  linspace 必定会包含数据起点和终点，那么其步长则为(1-0) / 9 = 0.11111111。如果需要产生 0.1, 0.2, ... 1.0 这样的数据，只需要将数据起点 0 修改为 0.1 即可。
 除了上面介绍到的 arange 和 linspace，NumPy还提供了 logspace 函数，该函数使用方法与 linspace 使用方法一样，读者不妨自己动手试一下。

> 另见这些API
>
> [`array`](https://numpy.org/devdocs/reference/generated/numpy.array.html#numpy.array)， [`zeros`](https://numpy.org/devdocs/reference/generated/numpy.zeros.html#numpy.zeros)[ ](https://numpy.org/devdocs/reference/generated/numpy.zeros.html#numpy.zeros)， [`zeros_like`](https://numpy.org/devdocs/reference/generated/numpy.zeros_like.html#numpy.zeros_like)， [`ones`](https://numpy.org/devdocs/reference/generated/numpy.ones.html#numpy.ones)， [`ones_like`](https://numpy.org/devdocs/reference/generated/numpy.ones_like.html#numpy.ones_like)， [`empty`](https://numpy.org/devdocs/reference/generated/numpy.empty.html#numpy.empty)， [`empty_like`](https://numpy.org/devdocs/reference/generated/numpy.empty_like.html#numpy.empty_like)， [`arange`](https://numpy.org/devdocs/reference/generated/numpy.arange.html#numpy.arange)， [`linspace`](https://numpy.org/devdocs/reference/generated/numpy.linspace.html#numpy.linspace)， [`numpy.random.mtrand.RandomState.rand`](https://numpy.org/devdocs/reference/random/generated/numpy.random.mtrand.RandomState.rand.html#numpy.random.mtrand.RandomState.rand)， [`numpy.random.mtrand.RandomState.randn`](https://numpy.org/devdocs/reference/random/generated/numpy.random.mtrand.RandomState.randn.html#numpy.random.mtrand.RandomState.randn)， [`fromfunction`](https://numpy.org/devdocs/reference/generated/numpy.fromfunction.html#numpy.fromfunction)， [`fromfile`](https://numpy.org/devdocs/reference/generated/numpy.fromfile.html#numpy.fromfile)

## NumPy的算术运算

### 对应元素相乘

对应元素相乘（element-wise product）是两个矩阵中对应元素乘积。np.multiply 函数用于数组或矩阵对应元素相乘，输出与相乘数组或矩阵的大小一致，其格式如下:

```python
numpy.multiply(x1, x2, /, out=None, *, where=True,casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
```

其中x1，x2之间的对应元素相乘遵守广播规则，NumPy的广播规则本章第7小节将介绍。以下我们通过一些示例来进一步说明。

```python
A = np.array([[1, 2], [-1, 4]])
B = np.array([[2, 0], [3, 4]])
A*B
##结果如下：
array([[ 2,  0],
       [-3, 16]])
#或另一种表示方法
np.multiply(A,B)
#运算结果也是
array([[ 2,  0],
       [-3, 16]])
```

### 点积运算

点积运算（dot product）又称为内积，在NumPy用np.dot表示，其一般格式为：

```python
numpy.dot(a, b, out=None)
```

以下通过一个示例来说明dot的具体使用及注意事项。

```python
X1=np.array([[1,2],[3,4]])
X2=np.array([[5,6,7],[8,9,10]])
X3=np.dot(X1,X2)
print(X3)
print(X1@X2)
```

### 线性代数

```python
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)
print(a.transpose())
print(np.linalg.inv(a)) # 矩阵求逆
print(np.eye(2)) # unit 2x2 matrix; "eye" represents "I" 单位矩阵
print(np.trace(a))  # trace

y = np.array([[5.], [7.]])
np.linalg.solve(a, y) #矩阵的解 ax=y x的解
np.linalg.eig(j) # 矩阵特征向量
```



## 数组变形

在机器学习以及深度学习的任务中，通常需要将处理好的数据以模型能接受的格式喂给模型，然后模型通过一系列的运算，最终返回一个处理结果。然而，由于不同模型所接受的输入格式不一样，往往需要先对其进行一系列的变形和运算，从而将数据处理成符合模型要求的格式。最常见的是矩阵或者数组的运算，经常会遇到需要把多个向量或矩阵按某轴方向合并，或需要展平（如在卷积或循环神经网络中，在全连接层之前，需要把矩阵展平）。下面介绍几种常用数据变形方法。

### 更改数组的形状

修改指定数组的形状是 NumPy 中最常见的操作之一，常见的方法有很多，下表列出了一些常用函数。NumPy中改变向量形状的一些函数：

<div align="center"><img src="chapter8/res/chapter8-4.png"></div>

```python
>>> a = np.floor(10*np.random.random((3,4)))
>>> a
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
>>> a.shape
(3, 4)
```

### 合并数组

<div align="center"><img src="chapter8/res/chapter8-5.png"></div>

（1）append
 合并一维数组

```python
import numpy as np
 
a =np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.append(a, b)
print(c) 
# [1 2 3 4 5 6]
```

合并多维数组

```python
import numpy as np
 
a =np.arange(4).reshape(2, 2)
b = np.arange(4).reshape(2, 2)
# 按行合并
c = np.append(a, b, axis=0)
print('按行合并后的结果')
print(c)
print('合并后数据维度', c.shape)
# 按列合并
d = np.append(a, b, axis=1)
print('按列合并后的结果')
print(d)
print('合并后数据维度', d.shape)
```

（2）concatenate
 沿指定轴连接数组或矩阵

```python
import numpy as np
a =np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
 
c = np.concatenate((a, b), axis=0)
print(c)
d = np.concatenate((a, b.T), axis=1)
print(d)
```

（3）stack
 沿指定轴堆叠数组或矩阵

```python
import numpy as np
 
a =np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.stack((a, b), axis=0))
```



### 数组的性质

- **ndarray.ndim** - 数组的轴（维度）的个数。在Python世界中，维度的数量被称为rank。

- **ndarray.shape** - 数组的维度。这是一个整数的元组，表示每个维度中数组的大小。对于有 *n* 行和 *m* 列的矩阵，`shape` 将是 `(n,m)`。因此，`shape` 元组的长度就是rank或维度的个数 `ndim`。

- **ndarray.size** - 数组元素的总数。这等于 `shape` 的元素的乘积。

- **ndarray.dtype** - 一个描述数组中元素类型的对象。可以使用标准的Python类型创建或指定dtype。另外NumPy提供它自己的类型。例如numpy.int32、numpy.int16和numpy.float64。

- **ndarray.itemsize** - 数组中每个元素的字节大小。例如，元素为 `float64` 类型的数组的 `itemsize` 为8（=64/8），而 `complex32` 类型的数组的 `itemsize` 为4（=32/8）。它等于 `ndarray.dtype.itemsize` 。

- **ndarray.data** - 该缓冲区包含数组的实际元素。通常，我们不需要使用此属性，因为我们将使用索引访问数组中的元素。

```python
>>> import numpy as np
>>> a = np.arange(15).reshape(3, 5)
>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> a.shape
(3, 5)
>>> a.ndim
2
>>> a.dtype.name
'int64'
>>> a.itemsize
8
>>> a.size
15
>>> type(a)
<type 'numpy.ndarray'>
>>> b = np.array([6, 7, 8])
>>> b
array([6, 7, 8])
>>> type(b)
<type 'numpy.ndarray'>
```

  

## 批量处理

在深度学习中，由于源数据都比较大，所以通常需要采用批处理。如利用批量来计算梯度的随机梯度法（SGD），就是一个典型应用。深度学习的计算一般比较复杂，加上数据量一般比较大，如果一次处理整个数据，往往出现资源瓶颈。为了更有效的计算，一般将整个数据集分成小批量。与处理整个数据集的另一个极端是每次处理一条记录，这种方法也不科学，一次处理一条记录无法充分发挥GPU、NumPy平行处理优势。因此，实际使用中往往采用批量处理（mini-batch）。
 如何把大数据拆分成多个批次呢？可采用如下步骤：
 （1）得到数据集
 （2）随机打乱数据
 （3）定义批大小
 （4）批处理数据集

```python
import numpy as np
#生成10000个形状为2X3的矩阵
data_train = np.random.randn(10000,2,3)
#这是一个3维矩阵，第一个维度为样本数，后两个是数据形状
print(data_train.shape)
#(10000,2,3)
#打乱这10000条数据
np.random.shuffle(data_train)
#定义批量大小
batch_size=100
#进行批处理
for i in range(0,len(data_train),batch_size):
    x_batch_sum=np.sum(data_train[i:i+batch_size])
    print("第{}批次,该批次的数据之和:{}".format(i,x_batch_sum))
```

## 通用函数

<div align="center"><img src="chapter8/res/chapter8-6.png"></div>

math与numpy函数的性能比较：

```python
import time
import math
import numpy as np
 
x = [i * 0.001 for i in np.arange(1000000)]
start = time.time()
for i, t in enumerate(x):
    x[i] = math.sin(t)
print ("math.sin:", time.time() - start )
 
x = [i * 0.001 for i in np.arange(1000000)]
x = np.array(x)
start = time.time()
np.sin(x)
print ("numpy.sin:", time.time() - start )
```

```
math.sin: 0.45232319831848145
numpy.sin: 0.017247676849365234
```

## 拷贝和视图

当计算和操作数组时，有时会将数据复制到新数组中，有时则不会。这通常是初学者混淆的根源。有三种情况：

### 完全不复制

简单分配不会复制数组对象或其数据。

```python
>>> a = np.arange(12)
>>> b = a            # no new object is created
>>> b is a           # a and b are two names for the same ndarray object
True
>>> b.shape = 3,4    # changes the shape of a
>>> a.shape
(3, 4)
```

Python将可变对象作为引用传递，因此函数调用不会复制。

```python
>>> def f(x):
...     print(id(x))
...
>>> id(a)                           # id is a unique identifier of an object
148293216
>>> f(a)
148293216
```

### 视图或浅拷贝

不同的数组对象可以共享相同的数据。该`view`方法创建一个查看相同数据的新数组对象。

```python
>>> c = a.view()
>>> c is a
False
>>> c.base is a                        # c is a view of the data owned by a
True
>>> c.flags.owndata
False
>>>
>>> c.shape = 2,6                      # a's shape doesn't change
>>> a.shape
(3, 4)
>>> c[0,4] = 1234                      # a's data changes
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
```

切片数组会返回一个视图：

```python
>>> s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
>>> s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

### 深拷贝

该`copy`方法生成数组及其数据的完整副本。

```python
>>> d = a.copy()                          # a new array object with new data is created
>>> d is a
False
>>> d.base is a                           # d doesn't share anything with a
False
>>> d[0,0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

有时，如果不再需要原始数组，则应在切片后调用 `copy`。例如，假设a是一个巨大的中间结果，最终结果b只包含a的一小部分，那么在用切片构造b时应该做一个深拷贝：

```python
>>> a = np.arange(int(1e8))
>>> b = a[:100].copy()
>>> del a  # the memory of ``a`` can be released.
```

如果改为使用 `b = a[:100]`，则 `a` 由 `b` 引用，并且即使执行 `del a` 也会在内存中持久存在。

## 广播机制

NumPy的Universal functions 中要求输入的数组shape是一致的，当数组的shape不相等的时候，则会使用广播机制。不过，调整数组使得shape一样，需满足一定规则，否则将出错。这些规则可归结为以下四条：

- 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分都通过在前面加1补齐；
   如：a：2x3x2 b：3x2，则b向a看齐，在b的前面加1：变为：1x3x2
- 输出数组的shape是输入数组shape的各个轴上的最大值；
- 如果输入数组的某个轴和输出数组的对应轴的长度相同或者其长度为1时，这个数组能够用来计算，否则出错；
- 当输入数组的某个轴的长度为1时，沿着此轴运算时都用（或复制）此轴上的第一组值。

<div align="center"><img src="chapter8/res/chapter8-7.png"></div>

## 花式索引和索引技巧

NumPy提供比常规Python序列更多的索引功能。除了通过整数和切片进行索引之外，正如我们之前看到的，数组可以由整数数组和布尔数组索引。

### 使用索引数组进行索引

```python
>>> a = np.arange(12)**2                       # the first 12 square numbers
>>> i = np.array( [ 1,1,3,8,5 ] )              # an array of indices
>>> a[i]                                       # the elements of a at the positions i
array([ 1,  1,  9, 64, 25])
>>>
>>> j = np.array( [ [ 3, 4], [ 9, 7 ] ] )      # a bidimensional array of indices
>>> a[j]                                       # the same shape as j
array([[ 9, 16],
       [81, 49]])
```

当索引数组`a`是多维的时，单个索引数组指的是第一个维度`a`。以下示例通过使用调色板将标签图像转换为彩色图像来显示此行为。

```python
>>> palette = np.array( [ [0,0,0],                # black
...                       [255,0,0],              # red
...                       [0,255,0],              # green
...                       [0,0,255],              # blue
...                       [255,255,255] ] )       # white
>>> image = np.array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette
...                     [ 0, 3, 4, 0 ]  ] )
>>> palette[image]                            # the (2,4,3) color image
array([[[  0,   0,   0],
        [255,   0,   0],
        [  0, 255,   0],
        [  0,   0,   0]],
       [[  0,   0,   0],
        [  0,   0, 255],
        [255, 255, 255],
        [  0,   0,   0]]])
```

### 使用布尔数组进行索引

当我们使用（整数）索引数组索引数组时，我们提供了要选择的索引列表。使用布尔索引，方法是不同的; 我们明确地选择我们想要的数组中的哪些项目以及我们不需要的项目。

人们可以想到的最自然的布尔索引方法是使用与原始数组具有 *相同形状的* 布尔数组：

```python
>>> a = np.arange(12).reshape(3,4)
>>> b = a > 4
>>> b                                          # b is a boolean with a's shape
array([[False, False, False, False],
       [False,  True,  True,  True],
       [ True,  True,  True,  True]])
>>> a[b]                                       # 1d array with the selected elements
array([ 5,  6,  7,  8,  9, 10, 11])
```

此属性在分配中非常有用：

```python
>>> a[b] = 0                                   # All elements of 'a' higher than 4 become 0
>>> a
array([[0, 1, 2, 3],
       [4, 0, 0, 0],
       [0, 0, 0, 0]])
```

使用布尔值进行索引的第二种方法更类似于整数索引; 对于数组的每个维度，我们给出一个1D布尔数组，选择我们想要的切片：

```python
>>> a = np.arange(12).reshape(3,4)
>>> b1 = np.array([False,True,True])             # first dim selection
>>> b2 = np.array([True,False,True,False])       # second dim selection
>>>
>>> a[b1,:]                                   # selecting rows
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> a[b1]                                     # same thing
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> a[:,b2]                                   # selecting columns
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10]])
>>>
>>> a[b1,b2]                                  # a weird thing to do
array([ 4, 10])
```

请注意，1D布尔数组的长度必须与要切片的尺寸（或轴）的长度一致。在前面的例子中，`b1`具有长度为3（的数目 *的行* 中`a`），和 `b2`（长度4）适合于索引的第二轴线（列） `a`。

### ix_()函数

[`ix_`](https://numpy.org/devdocs/reference/generated/numpy.ix_.html#numpy.ix_)函数可用于组合不同的向量，以便获得每个n-uplet的结果。例如，如果要计算从每个向量a，b和c中取得的所有三元组的所有a + b * c：

```python
>>> a = np.array([2,3,4,5])
>>> b = np.array([8,5,4])
>>> c = np.array([5,4,6,8,3])
>>> ax,bx,cx = np.ix_(a,b,c)
>>> ax
array([[[2]],
       [[3]],
       [[4]],
       [[5]]])
>>> bx
array([[[8],
        [5],
        [4]]])
>>> cx
array([[[5, 4, 6, 8, 3]]])
>>> ax.shape, bx.shape, cx.shape
((4, 1, 1), (1, 3, 1), (1, 1, 5))
>>> result = ax+bx*cx
>>> result
array([[[42, 34, 50, 66, 26],
        [27, 22, 32, 42, 17],
        [22, 18, 26, 34, 14]],
       [[43, 35, 51, 67, 27],
        [28, 23, 33, 43, 18],
        [23, 19, 27, 35, 15]],
       [[44, 36, 52, 68, 28],
        [29, 24, 34, 44, 19],
        [24, 20, 28, 36, 16]],
       [[45, 37, 53, 69, 29],
        [30, 25, 35, 45, 20],
        [25, 21, 29, 37, 17]]])
>>> result[3,2,4]
17
>>> a[3]+b[2]*c[4]
17
```

您还可以按如下方式实现reduce：

```python
>>> def ufunc_reduce(ufct, *vectors):
...    vs = np.ix_(*vectors)
...    r = ufct.identity
...    for v in vs:
...        r = ufct(r,v)
...    return r
```

然后将其用作：

```python
>>> ufunc_reduce(np.add,a,b,c)
array([[[15, 14, 16, 18, 13],
        [12, 11, 13, 15, 10],
        [11, 10, 12, 14,  9]],
       [[16, 15, 17, 19, 14],
        [13, 12, 14, 16, 11],
        [12, 11, 13, 15, 10]],
       [[17, 16, 18, 20, 15],
        [14, 13, 15, 17, 12],
        [13, 12, 14, 16, 11]],
       [[18, 17, 19, 21, 16],
        [15, 14, 16, 18, 13],
        [14, 13, 15, 17, 12]]])
```

与普通的ufunc.reduce相比，这个版本的reduce的优点是它利用了广播规则 ，以避免创建一个参数数组，输出的大小乘以向量的数量。

>参考资料
>
>1. [Numpy中文网](https://www.numpy.org.cn/)
>
>2. [Python深度学习基于PyTorch](http://www.feiguyunai.com/index.php/2020/11/24/python-dl-baseon-pytorch-01/#1_NumPy)

