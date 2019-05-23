# ---author：星雨---#
# code: UTF-8 #
# function:  感知器算法
import  numpy as np
import matplotlib.pyplot as plt

#######一下为  输入数据 ############
global w,O,e,x1,d
i=0
w = np.array([0, 0])#参数：权重
O=0#偏置
e = [0]

x1 = np.array([[2,2],[1,-2],[-1,2],[-1,0]])
print(x1[:,0],x1[:,1]) #提取并验证第一列（x）&第二列(y)
xMix=np.min(x1[:,0]) #求出x中最小 的数可视化要用
xMax=np.max(x1[:,0])
yMix=np.min(x1[:,1])
yMax=np.max(x1[:,1])
num=x1.shape[0]
d = np.array([0,1,0,1])
x2 = x1.transpose() # 转变成列向量
# print("x2:=",x2)
# c=np.dot(w,x2)  #对于一维向量为：点积。
# x4=np.array(input("输入矩阵类型的数据 [x1,x2,d]"))

def step_function(x):
    # 稍微修改的阶跃函数
    return np.array(x >=0, dtype=np.int)
def perceptor(i,w,O,e):##感知器调参算法
    for i in range(x1.shape[0]):
        a = np.dot(w, x1[i]) + O
        y = step_function(a)  # 阶跃函数
        e.append(d[i] - y)
        print("第{}次 a:{} y:{} e:{} ".format(i + 1, a, y, e[i]))
        if e[i + 1] == 0:
            w = w
            O = O
            print("w={} O={} ".format(w, O, ))
        else:  ##调参数
            w = w + e[i + 1] * x1[i]
            O = O + e[i + 1]
            print("w={} O={} i={}".format(w, O, i))
    return i,w,O,e

def check(i,w,x1,e):##算出后验证的算法
    if i==(num-1): #反馈验证
        a = np.dot(w, x1[0]) + O #返回去用第一个参数验证
        y=step_function(a)
        e.append(d[0] - y)
        print("e:",e)
        if e[i]!=0: #如果不等于0 继续迭代
            perceptor(i, w, O, e)
        if e[i]==0:#偏差为0
            print("The final w={} O={}".format(w,O))
            x = np.arange(xMix-2, xMax+1,0.5)#xMin和xMax的用武之地
            y = -1*(x * w[0] + O) / w[1]
            s=(-1*w[0])/w[1] #求斜率
            b=-O/w[1]#截距
            plt.scatter(x2[0], x2[1])  # 画出散点图
            plt.title('Scatter image', fontsize=24)
            plt.xlabel("x", fontsize=24)
            plt.ylabel("y", fontsize=24)
            plt.plot(x, y, "r",label="y = {}x+({})".format(s,b),linewidth=2)
            plt.legend(loc=2)#loc:1右上角，2 左上角
            plt.grid(True)##允许出网格
            plt.show()
i,w,O,e=perceptor(i,w,O,e)
check(i, w, x1, e)











