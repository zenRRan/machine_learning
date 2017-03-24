import numpy as np
import random as rand
import math

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
    # print(X[0])
    # print(sum_x_p.shape)
    for i in range(N):
        sum_x_p = np.add(sum_x_p, X[i] * p[i][k])
        # print("X[i] \n",X[i],"p[i][k] \n",p[i][k],"X[i] * p[i][k] \n",X[i] * p[i][k])
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
    # print(g)
    f = int(g)
    b = pow(math.e, f)

    # gua = 1/(SIGMA[k]*pow(2*math.pi,0.5)) * pow(math.e, int(-1*(np.matmul(np.add(xi,-np.matrix(MU[k])), np.add(xi,-np.matrix(MU[k])).transpose())/(2*SIGMA[k]*SIGMA[k]))))
    return a * b
def P(i,k):
    a = PI[k]*guass(X[i],k)
    b = 0.
    for kk in range(len(x1)):
        b +=  PI[k]*guass(X[i],kk)
    # print("a=",a,"b=",b)
    return a/b

x1 = [1,2,3,1,2,3]
x2 = [1,1,1,2,2,3]

num = 5
meu = 0.2
rand = rand.Random()
rand.seed(0)
steps = 100
random_x = []
random_y = []

for i in range(len(x1)):
    X.extend(rand.normalvariate(meu, x1[i]) for _ in range(num))
    X.extend(rand.normalvariate(meu, x2[i]) for _ in range(num))

N = len(X)//2

#init para
MU = X[:len(x1)]
MU.extend(X[N//2:N//2+len(x1)])
MU = np.matrix(MU).reshape([len(x1),2])
X = np.matrix(X).reshape([N,2])
p = np.zeros([N, len(x1)])
for i in range(N):
    for j in range(len(x1)):
        p[i][j] = 1 / len(x1)
PI = np.zeros(len(x1))
for i in range(len(x1)):
    PI[i] = 1/len(x1)
SIGMA = np.ones([len(x1),1])
print("p:\n",p)
print("PI:\n",PI)
print("MU:\n",MU)
print("SIGMA:\n",SIGMA)

a = np.matrix([1,2,3,4])
b = np.matrix([2,3])
# print([[2,2]]-3)

for step in range(steps):
    print("step ",step,"\n",p)
    print("MU ",MU)
    for i in range(N):
        # print("@")
        # print(p)
        for k in range(len(x1)):
            # print("@@")
            p[i][k] = P(i, k)
    for k in range(len(x1)):
        # print("#")
        PI[k] = pai(k)
        MU[k] = mu(k)
        SIGMA[k] = sigma(k)

