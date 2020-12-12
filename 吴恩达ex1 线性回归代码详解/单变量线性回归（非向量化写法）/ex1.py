from matplotlib import pyplot as plt
import pandas as pd
import random
import numpy as np
#导入包 pyplot是用来画图的 pandas用来读取数据处理数据的
#      random用来生成随机数 numpy是用来做矩阵运算的
#      这一版代码中 用的是非向量化的方式 所以导入了numpy但是没有用到
#首先我们了解一下步骤：
#step1：假设函数：hx=theta0 + theta1 * x 并初始化theta0与theta1
#step2：cost函数：（hx-y）^2/ 求m项的和 再除以2m m是指数据的条数
#step3：如果cost函数求出来的值 也就是model error模型误差太大 就进行梯度下降
#step4：梯度下降：theta0=theta0-学习率*导数项
#               theta1=theta1-学习率*导数项
#               导数项为 （hx-y）*xi 求m项的和 并除以m  m是指数据的条数
#               对于theta0来说 相当于theta0*x0 x0=1 所以导数项就等于
#               （hx-y）*x0 ---> （hx-y）*1
#               对于theta1来说 导数项就等于
#               （hx-y）*x
#所以梯度下降的公式可以写成：theta0=theta0-学习率*（hx-y）*1
#                      theta1=theta1-学习率*（hx-y）*x
#然后重复进行step1 2 3 4 直至收敛
#上面的就是进行线性回归的基本流程 下面非向量化代码的详细解释
#由于是我第一次写读取文件的代码 过程很繁琐 显得很蠢 在向量化代码中改的很简单了
df = pd.read_table("ex1data1.txt")#使用pandas的readtable的方法读取txt文件
features = []#创建一个features列表用来存储数据
#!!!!这里需要注意 因为原始数据中 没有列名 而且数据之间使用“，”隔开，当时我不知道读取数据时可以加上列名，所以手动在txt文件开头加了一行 features
#在向量化代码中 读取数据会变得更简单
for row in df.itertuples(index=True, name='Pandas'):
    features.append(getattr(row, "features"))  #遍历数据的每一行，将列名为features的数据加入features列表中

# print(features)
x1 = []
x2 = []#创建两个列表用来存储使用“，”分开的数据  其中x1对应着上面线性回归流程中的x x2对应着y
for elements in features:
    elements = elements.split(",")
    # print(elements[0],elements[1])
    x1.append(float(elements[0]))
    x2.append(float(elements[1]))
#遍历features中的每个元素 其实每个元素的形式为“x1，x2”如“5.11，65.25”这样
#使用split按照“，”将每个元素分成两部分 第一部分存入x1列表 第二部分存入x2列表
plt.scatter(x1, x2)#使用散点图画出数据的分布
plt.show()
#到此我们就取出了x1 x2 并把他们存入了不同的列表以备后用 下面就开始线性回归的步骤啦
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
theta0, theta1,= random.randrange(-1, 1), random.randrange(-1, 1)#初始化theta0，theta1，机器学习课程中讲的也可以设置为0
x0 = [1 for i in range(97)]#其实上面提取数据只提取了x 对于theta1而言也就是x1 还有真实值y 但 对于theta0来说 虽然式子中只有theta0 其实相当于都乘了一个x0
#只不过x0都是1 所以这里也用了一个列表来表示x0
modelError = 0#设置模型误差 以备后用 其实就是（hx-y）^2 后面会循环m次 累加
learningRate = 0.001#设置学习率
sumModelError = 1000#设置总的模型误差 也就是代价 是上面累加完的modelError 然后再除以2m
lastSumModelError=10000#设置上一次的模型误差 可能你不懂这是用来干什么的 后面会详细解释的 先跳过这个变量也可以
derivative0=0 #设置theta0的导数项为0 用于后面进行累加
derivative1=0 #设置theta1的导数项为0 用于后面进行累加
#上面我设置了很多变量为0 用于累加 如果没有编程经验的可能不懂 就像求和需要设置一个sum=0 然后sum+=i一样 这是一种很小白的做法 可是对于第一次写线性回归我只会这么做了 哈哈

sumModelError_arr=[]#设置总的模型误差 也就是总的代价 总的损失（如果只听了机器学习这一门课可能不知道什么是损失，再继续听吴恩达的深度学习就懂啦）用于存入迭代过程中的代价，用于以后画图
derivative0_arr=[]#设置theta0的导数项的数组用于存入迭代过程中的导数项，用于以后画图
derivative1_arr=[]#设置theta1的导数项的数组用于存入迭代过程中的导数项，用于以后画图
minus=1#这是设置的一个变量用来判断迭代到底该什么时候停止 minus=上一次的总的代价-这一次的总的代价
for j in range(0,10000000):#先设置一个很大很大的迭代次数 这里设置了10000000
    if minus>0.0001:#这个判断的意思是 如果上一次的总的代价-这一次的总的代价 >0.0001 就继续迭代 如果小于等于 就说明已经收敛了 就不停止迭代了
        for i in range(0, len(x0)):#循环m次 m是数据条数 相当于每次都按顺序取不同的数据 第一条 第二条
            derivative0 += ((theta0 * x0[i] + theta1 * x1[i]) - x2[i]) * x0[i]#theta0的导数项 还记得我们上面流程中的公式么？
            #               对于theta0来说 相当于theta0*x0 x0=1 所以导数项就等于
            #               （hx-y）*x0 ---> （hx-y）*1
            derivative1 += ((theta0 * x0[i] + theta1 * x1[i]) - x2[i]) * x1[i]#theta1的导数项
            #               对于theta1来说 导数项就等于
            #               （hx-y）*x
            modelError += ((theta0 * x0[i] + theta1 * x1[i]) - x2[i]) *((theta0 * x0[i] + theta1 * x1[i]) - x2[i])
            #每一项的误差 代价 损失 这三种叫法都可以 然后累加
            #（hx-y）^2/ 求m项的和

        derivative0 = derivative0 / len(x0)#导数项还需除以m
        # print(j, i, derivative0)
        derivative1 = derivative1 / len(x0)#导数项还需除以m
        # print(j, i, derivative1)
        #导数项为 （hx-y）*xi 求m项的和 并除以m  m是指数据的条数
        temp0 = theta0 - learningRate * derivative0#计算新的theta0
        # print("derivative1", derivative1)
        temp1 = theta1 - learningRate * derivative1#计算新的theta1
        #梯度下降：theta0 = theta0 - 学习率 * 导数项
        #               theta1=theta1-学习率*导数项
        #之所以使用temp0 temp1是因为需要同步更新 我记得当时写的时候是按照吴恩达课程上给的伪代码写的 现在回过头写注释 感觉直接赋值给theta0 theta1好像也可以
        theta0 = temp0
        theta1 = temp1
        # print(theta0,theta1)
        sumModelError = modelError / (2 * len(x0))#计算总的代价 就多了一个除以2m
        sumModelError1=lastSumModelError
        minus = sumModelError1 - sumModelError
        lastSumModelError=sumModelError
        modelError=0# 上面四行代码都是为了计算上一次的代价-下一次的代价所做的准备 主要是逻辑关系有点难想明白 你按照上面四行去赋值 自己思考一下顺序就懂啦
        # print("derivative0",derivative0,"derivative1",derivative1)
        # print("sumModelError",sumModelError)
        sumModelError_arr.append(sumModelError)#把这一次的总代价加入到列表中
        derivative0_arr.append(derivative0)#把这一次theta0的导数项加入到列表中
        derivative1_arr.append(derivative1)#把这一次theta1的导数项加入到列表中
        #print(minus)
        #看到这里 你已经完成了一次循环过程啦 如果到这里minus（我们上面用四行代码设置的）还满足>0.0001这个要求的话 就会一直循环迭代下去
        #如果不满足 就会跳到下面的else中 break 停止循环啦
    else:
        break
plt.plot(sumModelError_arr,color="blue")
plt.show()#画图 画的图就是总的损失 或者说 总的代价随迭代次数的变化 最后会看到会收敛到一个值 然后线性回归就大功告成啦

#这是我的第一次写关于机器学习的文章 有什么不足请评论区留言啦 代码很繁琐 在下一篇 单变量线性回归（向量化）的代码中会很变得很简洁的啦

