# --coding:utf-8--

import numpy as np
from numpy import *
from scipy.spatial import distance
from sklearn.metrics.pairwise import haversine_distances


def get_distance(vector1, vector2, distance_type='ou', p=10):
    """
    距离度量用于计算给定问题空间中两个对象之间的差异，即数据集中的特征。 然后可以使用该距离来确定特征之间的相似性， 距离越小特征越相似。
    对于距离的度量，我们可以在几何距离测量和统计距离测量之间进行选择，应该选择哪种距离度量取决于数据的类型。 特征可能有不同的数据类型（例如，真实值、布尔值、分类值），数据可能是多维的或由地理空间数据组成。
    https://baijiahao.baidu.com/s?id=1747898995916547028&wfr=spider&for=pc
    """

    print("distance_type:" + str(distance_type))

    if distance_type == 'ou':
        """
        1.欧式距离(Euclidean distance). 欧氏距离度量两个实值向量之间的最短距离。由于其直观，使用简单和对许多用例有良好结果，所以它是最常用的距离度量和许多应用程序的默认距离度量。
        欧氏距离有两个主要缺点。首先，距离测量不适用于比2D或3D空间更高维度的数据。第二，如果我们不将特征规范化和/或标准化，距离可能会因为单位的不同而倾斜。
        """
        # ou = np.sqrt(np.sum(np.square(vector1 - vector2)))
        ou = distance.euclidean(vector1, vector2)
        print('欧氏距离：', ou)
        return ou
    elif distance_type == 'man':
        """
        2.曼哈顿距离(Manhattan distance). 曼哈顿距离也被称为出租车或城市街区距离，因为两个实值向量之间的距离是根据一个人只能以直角移动计算的。这种距离度量通常用于离散和二元属性，这样可以获得真实的路径。
        曼哈顿的距离有两个主要的缺点。它不如高维空间中的欧氏距离直观，它也没有显示可能的最短路径。虽然这可能没有问题，但我们应该意识到这并不是最短的距离。
        """
        # manhadun = sum(abs(vector1 - vector2))
        manhadun = distance.cityblock(vector1, vector2)
        print("曼哈顿距离：", manhadun)
        return manhadun
    elif distance_type == 'qbxf':
        """
        3.切比雪夫距离(Chebyshev distance). 切比雪夫距离也称为棋盘距离，因为它是两个实值向量之间任意维度上的最大距离。 它通常用于仓库物流中，其中最长的路径决定了从一个点到另一个点所需的时间。
        切比雪夫距离只有非常特定的用例，因此很少使用。
        """
        # qbxf = abs(vector1 - vector2).max()
        qbxf = distance.chebyshev(vector1, vector2)
        print("切比雪夫距离：", qbxf)
        return qbxf
    elif distance_type == 'min':
        """
        4.闵可夫斯基距离(Minkowski distance). 闵可夫斯基距离是上述距离度量的广义形式。 它可以用于相同的用例，同时提供高灵活性。 我们可以选择 p 值来找到最合适的距离度量。
        由于闵可夫斯基距离表示不同的距离度量，它就有与它们相同的主要缺点，例如在高维空间的问题和对特征单位的依赖。此外，p值的灵活性也可能是一个缺点，因为它可能降低计算效率，因为找到正确的p值需要进行多次计算。
        """
        distance.minkowski(vector1, vector2, p)
    elif distance_type == 'cos':
        """
        5.余弦相似度和距离(Cosine similarity). 余弦相似度是方向的度量，他的大小由两个向量之间的余弦决定，并且忽略了向量的大小。 余弦相似度通常用于与数据大小无关紧要的高维，例如，推荐系统或文本分析。
        余弦相似度常用于范围在0到1之间的正空间中。余弦距离就是用1减去余弦相似度，位于0(相似值)和1(不同值)之间。
        余弦距离的主要缺点是它不考虑大小而只考虑向量的方向。因此，没有充分考虑到值的差异。
        """
        # 4.夹角余弦距离
        # n1 = np.squeeze(np.asarray(vector1))
        # n2 = np.squeeze(np.asarray(vector2))
        # cos = dot(n1, n2) / (linalg.norm(n1) * linalg.norm(n2))
        cos = distance.cosine(vector1, vector2)
        print("夹角余弦距离：", cos)
        return cos
    elif distance_type == 'haver':
        """
        6.半正矢距离(Haversine distance). 半正矢距离测量的是球面上两点之间的最短距离。因此常用于导航，其中经度和纬度和曲率对计算都有影响。
        半正矢距离的主要缺点是假设是一个球体，而这种情况很少出现。
        """
        haver = haversine_distances([vector1, vector2])
        return haver
    elif distance_type == 'hamming':
        """
        7.汉明距离衡量两个二进制向量或字符串之间的差异. 对向量按元素进行比较，并对差异的数量进行平均。如果两个向量相同，得到的距离是0之间，如果两个向量完全不同，得到的距离是1。
        汉明距离有两个主要缺点。距离测量只能比较相同长度的向量，它不能给出差异的大小。所以当差异的大小很重要时，不建议使用汉明距离。
        """
        hamming = distance.hamming(vector1, vector2)
        return hamming
    elif distance_type == 'jaccard':
        """
        8.杰卡德指数和距离(Jaccard Index). Jaccard指数用于确定两个样本集之间的相似性。 它反映了与整个数据集相比存在多少一对一匹配。 Jaccard指数通常用于二进制数据比如图像识别的深度学习模型的预测与标记数据进行比较，或者根据单词的重叠来比较文档中的文本模式。
        Jaccard指数和距离的主要缺点是，它受到数据规模的强烈影响，即每个项目的权重与数据集的规模成反比。
        """
        jaccard = distance.jaccard(vector1, vector2)
        return jaccard
    elif distance_type == 'sorensen':
        """
        9.Sorensen-Dice指数. Srensen-Dice指数类似于Jaccard指数，它可以衡量的是样本集的相似性和多样性。该指数更直观，因为它计算重叠的百分比。Srensen-Dice索引常用于图像分割和文本相似度分析。
        它的主要缺点也是受数据集大小的影响很大。
        """
        sorensen = distance.dice(vector1, vector2)
        return sorensen
    elif distance_type == 'sorensen':
        """
        10.动态时间规整(Dynamic Time Warping).动态时间规整是测量两个不同长度时间序列之间距离的一种重要方法。可以用于所有时间序列数据的用例，如语音识别或异常检测。
        为什么我们需要一个为时间序列进行距离测量的度量呢？如果时间序列长度不同或失真，则上述面说到的其他距离测量无法确定良好的相似性。比如欧几里得距离计算每个时间步长的两个时间序列之间的距离。但是如果两个时间序列的形状相同但在时间上发生了偏移，那么尽管时间序列非常相似，但欧几里得距离会表现出很大的差异。

        动态时间规整通过使用多对一或一对多映射来最小化两个时间序列之间的总距离来避免这个问题。当搜索最佳对齐时，这会产生更直观的相似性度量。通过动态规划找到一条弯曲的路径最小化距离，该路径必须满足以下条件：

        边界条件:弯曲路径在两个时间序列的起始点和结束点开始和结束

        单调性条件:保持点的时间顺序，避免时间倒流

        连续条件:路径转换限制在相邻的时间点上，避免时间跳跃

        整经窗口条件(可选):允许的点落入给定宽度的整经窗口

        坡度条件(可选):限制弯曲路径坡度，避免极端运动
        
        动态时间规整的一个主要缺点是与其他距离测量方法相比，它的计算工作量相对较高。
        """
        # from fastdtw import fastdtw

        # distance, path = fastdtw(timeseries_1, timeseries_2, dist=euclidean)

        # return distance, path
    else:
        raise ValueError('Distance type {} is invalid.'.format(distance_type))




def Normalize(data):
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]
