import numpy as np
import random as rand
import math
import matplotlib.pyplot as plt

p = []
X = []
N = 0

def sum_p(k):
    sum = 1e-15
    for i in range(N):
        sum += p[i][k]
    return sum

def pai(k):
    return sum_p(k) / N

def mu(k):
    sum = sum_p(k)
    sum_x_p = np.zeros(shape=np.matrix(X[0]).shape)
    for i in range(N):
        sum_x_p = np.add(sum_x_p, X[i] * p[i][k])
    MU_X[k] = (sum_x_p / sum)[0][0]
    MU_Y[k] = (sum_x_p / sum)[0][1]
    return sum_x_p/sum

def sigma(k):
    sum = 1.
    for i in range(N):
        sum += p[i][k]*np.matmul(X[i]-mu(k), (X[i]-mu(k)).transpose())
    return sum/sum_p(k)

def guass(xi, k):
    # print(k)
    a = 1/(SIGMA[k]*pow(2*math.pi,0.5))
    c = np.add(xi,-np.matrix(MU[k]))
    e = 2*SIGMA[k]*SIGMA[k]
    g = -1*(np.matmul(c, c.transpose())/e)
    # print(np.matmul(c, c.transpose())/e)
    # print("g=", g)
    f = float(g)
    # print("f=",f)
    b = pow(math.e, f)
    # b = math.e ** g
    # gua = 1/(SIGMA[k]*pow(2*math.pi,0.5)) * pow(math.e, int(-1*(np.matmul(np.add(xi,-np.matrix(MU[k])), np.add(xi,-np.matrix(MU[k])).transpose())/(2*SIGMA[k]*SIGMA[k]))))
    return a * b
def P(i,k):
    a = PI[k]*guass(X[i],k)
    b = 1e-15
    for kk in range(len(x1)):
        b +=  PI[kk] * guass(X[i],kk)
    # print("a=",a,"b=",b)
    return a/b

x1 = [1,2,2]
x2 = [2,2,1]

num = 100
meu = 0.2
rand = rand.Random()
rand.seed(0)
steps = 20
random_x = []
random_y = []

colorSet = ['blue', 'green', 'red']

for i in range(len(x1)):
    random_x.extend(rand.normalvariate(x1[i], meu) for _ in range(num))
    random_y.extend(rand.normalvariate(x2[i], meu) for _ in range(num))

X = [[random_x[i],random_y[i]] for i in range(len(random_x))]

# plt.figure(figsize=(9, 6))
# plt.scatter(random_x, random_y, color='black')
# plt.scatter(x1, x2, color=colorSet, s=100)
# plt.show()
N = len(X)//2

#init para
MU = [[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]]
MU_X = [0.3, 0.4, 0.5]
MU_Y = [0.3, 0.4, 0.5]
p = [[0,0,0] for _ in range(len(random_x))]
# for i in range(N):
#     for j in range(len(x1)):
#         p[i][j] = 1 / len(x1)
PI = np.zeros(len(x1))
for i in range(len(x1)):
    PI[i] = 1/len(x1)
SIGMA = np.ones([len(x1),1])
print("p:\n",p)
print("PI:\n",PI)
print("MU:\n",MU)
print("SIGMA:\n",SIGMA)

def draw():
    plt.figure(figsize=(9, 6))
    colors = []
    for i in range(len(random_x)):
        colors.append(colorSet[p[i].index(max(p[i]))])
    # MU_X = [MU_XY[0] for MU_XY in MU]
    # MU_Y = [MU_XY[1] for MU_XY in MU]
    # print("MU_X=", MU_X)
    # print("MU_Y=", MU_Y)
    plt.scatter(MU_X, MU_Y, color=colorSet, s=100, alpha=0.7)
    plt.scatter(random_x, random_y, color=colors)
    plt.show()
draw()
for step in range(steps):
    print("############################################################################")
    print("step ",step)
    for i in range(N):
        for k in range(len(x1)):
            p[i][k] = P(i, k)
    # print("p=\n",p)
    for k in range(len(x1)):
        PI[k] = pai(k)
    # print("PI=\n", PI)
    for k in range(len(x1)):
        MU[k] = mu(k)
    # print("MU=\n", MU)
    for k in range(len(x1)):
        SIGMA[k] = sigma(k)
    # print("SIGMA=\n", SIGMA)

    draw()
