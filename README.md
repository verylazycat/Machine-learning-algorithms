### KNN

###### 介绍

K最近邻分类算法（K-Nearest Neihbor），是一个理论上比较成熟的算法，也是最简单的机器学习算法，主要思路：如果一个样品在特征空间k个最相似的样本中的大多数属于某一个类别，则该样本也属于这个类别。

###### 算法：

1.计算已知类别数据的点于当前点之间的距离

2.按照距离递增次序排序

3.选取与当前点距离最小的k个点

4.确定前k个点所在类别的概率

5.返回前k个点所出现的频率最高的类别作为当前点的预测分类

###### 欧几里得距离：

$$
d(p,q) = \sqrt{\sum_{i=1}^n(qi-pi)^2}
$$



###### 代码演示

```python
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#从MNIST数据集中筛选出5000条数据用作测试
train_X,train_Y = mnist.train.next_batch(5000)
#从MNIST数据集中筛选出200条数据用作测试
test_X,test_Y = mnist.test.next_batch(100)
train2_X = tf.placeholder("float",[None,784])
test2_X = tf.placeholder("float",[784])
#使用L1距离计算KNN距离计算
distance = tf.reduce_sum(tf.abs(tf.add(train2_X,tf.negative(test2_X))),reduction_indices=1)
#预测：取得最近的邻居节点
pred = tf.arg_min(distance,0)
accuracy = 0
#变量初始化
init = tf.global_variables_initializer()
#启动图
with tf.Session() as sess:
    sess.run(init)
    #遍历测试数据集
    for i in range(len(test_X)):
        #获取最近的邻居节点
        nn_index = sess.run(pred,feed_dict={train2_X:train_X,test2_X:test_X[i,:]})
        #获取最近的邻居节点的类别标签，并将其与该节点的真实类别标签进行比较
        print("测试数据",i,"预测分类:",np.argmax(train_Y[nn_index]),"真实类别:",np.argmax(test_Y[i]))
        #计算准确率
        if np.argmax(train_Y[nn_index]) == np.argmax(test_Y[i]):
            accuracy += 1./len(test_X)
    print("分类准确率为:",accuracy)
```

![](images\knn.PNG)

------

### Decising tree

###### 介绍：

决策树(Decision Tree）是在已知各种情况发生概率的[基础](https://baike.baidu.com/item/%E5%9F%BA%E7%A1%80/32794)上，通过构成决策树来求取净现值的[期望](https://baike.baidu.com/item/%E6%9C%9F%E6%9C%9B/35704)值大于等于零的概率，评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种图解法。由于这种决策分支画成图形很像一棵树的枝干，故称决策树。在机器学习中，决策树是一个预测模型，他代表的是对象属性与对象值之间的一种映射关系。Entropy= 系统的凌乱程度，使用算法[ID3](https://baike.baidu.com/item/ID3), [C4.5](https://baike.baidu.com/item/C4.5)和C5.0生成树算法使用熵。这一度量是基于信息学理论中熵的概念。分类树（决策树）是一种十分常用的分类方法。他是一种监管学习，所谓监管学习就是给定一堆样本，每个样本都有一组属性和一个类别，这些类别是事先确定的，那么通过学习得到一个分类器，这个分类器能够对新出现的对象给出正确的分类。这样的机器学习就被称之为监督学习。

##### 概念介绍：

###### 1.香农熵（entropy）：

 熵定义为信息的期望值。在信息论与概率统计中，熵是表示随机变量不确定性的度量。
$$
l(xi) = -log_2p(xi)
$$
p(xi)是选择该分类的概率

通过上式，可以计算所有类别的信息：
$$
H = -\sum_{i=1}^np(xi)log_2p(xi)
$$
n是分类数目，熵越大，随机变量的不确定性就越大

###### 2.信息增益：

如何选择特征，需要看信息增益。也就是说，信息增益是相对于特征而言的，信息增益越大，特征对最终的分类结果影响也就越大，我们就应该选择对最终分类结果影响最大的那个特征作为我们的分类特征。

明确一个概念，条件熵，条件熵H(Y|X)表示在已知随机变量X的条件下随机变量Y的不确定性，随机变量X给定的条件下随机变量Y的条件熵(conditional entropy) H(Y|X)，定义X给定条件下Y的条件概率分布的熵对X的数学期望：
$$
H（y|x） = \sum_{i=1}^np_iH(Y|X=x_i)
$$
特征A对训练数据集D的信息增益g(D,A)，定义为集合D的经验熵H(D)与特征A给定条件下D的经验条件熵H(D|A)之差，即
$$
g(D,A) = H(D)-H(D|A)
$$
即信息增益=经验熵 - 条件熵 

##### 算法：

决策树典型算法有ID3，C4.5，CART

此处介绍C4.5算法：

1.创建节点N

2.如果训练集为空，在节点N标记为Failure

3.如果训练集的所有记录都属于一个类别，则该类别标记节点N

4.如果候选属性为空，则返回N作为叶节点，标记为训练集中最普通的类

5.for each 候选属性 attribute_list

6.if 候选属性是连续的then

7.对该属性进行离散化

8.选择候选属性attribute_list中具有最高信息增益率的属性D

9.标记节点N为属性D

10.for each 属性D的一致值d

11.由[节点](https://baike.baidu.com/item/%E8%8A%82%E7%82%B9)N长出一个条件为D=d的分支

12.设s是训练集中D=d的训练样本的集合

13.if s为空

14.加上一个树叶，标记为训练集中最普通的类

15.else加上一个有C4.5（R - {D},C，s）返回的点

##### 代码演示：

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
print(iris)
iris_data=iris['data']
iris_lable = iris['target']
iris_target_name = iris['target_names']
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(iris_data,iris_lable)
print('[7,7,7,7]预测类别是',iris_target_name[clf.predict([[7,7,7,7]])])
```

![](images\Decising_tree_.PNG)

------

### SVM(支持向量机)

##### SVM介绍：

支持向量机（Support Vector Machine, SVM）是一类按[监督学习](https://baike.baidu.com/item/%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0/9820109)（supervised learning）方式对数据进行[二元分类](https://baike.baidu.com/item/%E4%BA%8C%E5%85%83%E5%88%86%E7%B1%BB/15635322)（binary classification）的广义线性分类器（generalized linear classifier），其[决策边界](https://baike.baidu.com/item/%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C/22778546)是对学习样本求解的最大边距超平面（maximum-margin hyperplane。SVM使用铰链损失函数（hinge loss）计算经验风险（empirical risk）并在求解系统中加入了正则化项以优化结构风险（structural risk），是一个具有稀疏性和稳健性的分类器 。SVM可以通过[核方法](https://baike.baidu.com/item/%E6%A0%B8%E6%96%B9%E6%B3%95/1683712)（kernel method）进行非线性分类，是常见的核学习（kernel learning）方法之一 。SVM被提出于1963年，在二十世纪90年代后得到快速发展并衍生出一系列改进和扩展算法，包括多分类SVM、最小二乘SVM（Least-Square SVM, LS-SVM）、支持向量回归（Support Vector Regression, SVR），支持向量聚类（support vector clustering）、半监督SVM（semi-supervised SVM, S3VM）等，在人像识别（face recognition）、文本分类（text categorization）等模式识别（pattern recognition）问题中有广泛应用。

###### 线性可分(linear separability)：

分类问题中给定输入数据和学习目标X={x1，x2...xn},y={y1,y2...yn},输入样品拥有多个特征，在几何上构成了特征空间（feature space）：X={x1...xn},Y={1-,1},即negative class和positive class。如果输入数据所在的特征空间存在决策边界（decision boundary）的超平面（hyperplane）：
$$
W^{T}+b=0
$$
将学习目标按照正负分类，并使任意样品到平面距离大于1：
$$
y_i(w^{T}X_i+b)>=1
$$
 则称该分类问题是具有线性可分性，w和b分别为超平面法向量和截距。

满足该条件的决策边界实际上构造了2个平行的超平面：
$$
w^{T}X_i+b-1>=+1,if y_i=+1
\\
w^{T}X_i+b-1<=+1,if y_i=-1
$$
两个间隔边距的距离：
$$
d=2/||w||
$$
![](images\linear separability.jpg)

###### 损失函数（loss function）：

在不具有线性可分性时，使用超平面作为决策边界会带来分类损失，即数据不再支持向量上，而是进入了间隔边界的内部，或者错误落入决策边界的一边。

损失函数可以对分类损失进行向量化：
$$
L(p)=\begin{cases}
0 & p<0 \\
1 & p>=0  \\
\end{cases}
$$

###### 其他损失函数：

###### hinge：

$$
L（p）=max(0,1,-p)
$$

###### logistic:

$$
L（P）=log[1+exp(-p)]
$$

###### exponential:

$$
L(p)=exp(-p)
$$

![](images\loss function.jpg)

###### 核方法：

一些线性不可分的问题可能时可分的，即特征空间存在超曲面（hypersurface）。使用非线性函数可以将非线性可分问题从原始的特征空间映射至更高维的希尔伯特空间H（Hilbert space）：
$$
w^{T}\phi(X)+b=0
$$

$$
\phi:x->H的映射函数
$$

###### 核函数：

$$
k(X_1,X_2)=\phi(X_1)^{T}\phi(X_2)
$$

![](images\核.png)

###### 代码：

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

# 加载数据
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])
# 分离训练和测试集
train_indices = np.random.choice(len(x_vals),
                                 round(len(x_vals)*0.8),
                                 replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
batch_size = 100

# 初始化feedin
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 创建变量
A = tf.Variable(tf.random_normal(shape=[2, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 定义线性模型
model_output = tf.subtract(tf.matmul(x_data, A), b)

# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(A))

# Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
alpha = tf.constant([0.01])
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(20000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = -a2/a1
y_intercept = b/a1
best_fit = []

x1_vals = [d[1] for d in x_vals]

for i in x1_vals:
    best_fit.append(slope*i+y_intercept)


# Separate I. setosa
setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]
not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]

plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()
```

![](images\SVM.png)

------

### 朴素贝叶斯

###### 介绍：

朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法 [1]  。最为广泛的两种分类模型是决策树模型(Decision Tree Model)和朴素贝叶斯模型（Naive Bayesian Model，NBM）。和决策树模型相比，**朴素贝叶斯分类器(Naive Bayes Classifier,或 NBC)**发源于古典数学理论，有着坚实的数学基础，以及稳定的分类效率。同时，NBC模型所需估计的参数很少，对缺失数据不太敏感，算法也比较简单。理论上，NBC模型与其他分类方法相比具有最小的误差率。但是实际上并非总是如此，这是因为NBC模型假设属性之间相互独立，这个假设在实际应用中往往是不成立的，这给NBC模型的正确分类带来了一定影响。

###### 贝叶斯：

$$
P(A|B)=P(A){P(B|A)/P(B)}
$$

P(A)是先验概率

P(A|B)是后验概率

P(B|A)/P(B)是可能性函数
$$
后验概率 = 先验概率*可能性函数
$$

###### 朴素贝叶斯：

朴素贝叶斯对条件概率分布做了条件独立性的假设。

###### 代码演示：

```python
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
if __name__ == '__main__':
    testingNB()
```

![](images\bayes.PNG)