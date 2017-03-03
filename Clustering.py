#每次plt.show()都会暂停 关闭图片继续执行

import matplotlib.pyplot as plt
from random import *

num = 100
train_steps = 100
mu = 0.2

random_x = []
random_y = []
y = []
rand = Random()
rand.seed(1) #seed() 方法改变随机数生成器的种子

#用正态分布作为随机参数
random_x.extend([rand.normalvariate(1, mu) for _ in range(num)])
random_x.extend([rand.normalvariate(1, mu) for _ in range(num)])
random_x.extend([rand.normalvariate(2, mu) for _ in range(num)])

random_y.extend([rand.normalvariate(1, mu) for _ in range(num)])
random_y.extend([rand.normalvariate(2, mu) for _ in range(num)])
random_y.extend([rand.normalvariate(2, mu) for _ in range(num)])

y.extend([0 for _ in range(num)])
y.extend([1 for _ in range(num)])
y.extend([2 for _ in range(num)])

#显示随机的300个点
# plt.scatter(random_x, random_y, color='black')
# plt.show()

#设置初始化的n个点 这个是分成n类的意思 这里是分成三类
pred_x = [0.2, 0.3, 0.4]
pred_y = [0.2, 0.3, 0.4]

#设置颜色
color_set = ['red','green','yellow']
plt.scatter(random_x, random_y, color='black')
plt.scatter(pred_x, pred_y, color=color_set, s=250, alpha=0.7) #s半径大小
plt.show()

label = []

#开始循环更新
for step in range(train_steps):
    print("Step=",step)
    for i in range(len(random_x)):
        min_j, min_dist = -1, -1
        for j in range(len(pred_x)):#在三个点中选择距离点
            dist = pow(random_x[i]-pred_x[j], 2) + pow(random_y[i]-pred_y[j], 2)
            if min_j == -1 or dist < min_dist:
               min_j, min_dist = j, dist
        if step == 0:
            label.append(min_j)
        else:
            label[i] = min_j
    for i in range(len(pred_x)):
        count, x_sum, y_sum = 1, pred_x[i], pred_y[i]
        for j in range(len(random_x)):
            if label[j] == i:
                count += 1
                x_sum += random_x[j]
                y_sum += random_y[j]
        pred_x[i], pred_y[i] = x_sum/count, y_sum/count

    #show更新的图
    pred_colors = [color_set[k] for k in label] #给点分类的颜色
    plt.scatter(random_x, random_y, color=pred_colors)
    plt.scatter(pred_x, pred_y, color=color_set, s=250, alpha=0.7)
    plt.show()
































