import numpy as np
import matplotlib as plt
'''
类的属性和类函数介绍：
属性：weight,bias,x,y
函数：更新weight函数,更新bias函数,判断是否有误分类点函数
'''
class Perceptron:
    def __init__(self,xtrain,ytrain,lr):
        self.l=lr
        self.x=xtrain
        self.y=ytrain
        self.n=xtrain.shape[1]
        self.row=xtrain.shape[0]
        self.w=np.zeros(self.n,dtype=np.int32)
        self.b=0
    '''
    计算误分类：循环所有x,y，计算y(w*x+b)并判断
    '''
    def mis_dot(self):
        i=0
        while True:
            if i<=self.row-1:
                # print(i, self.y[i]*(np.dot(self.w, self.x[i, :]) + self.b))
                if (self.y[i]*(np.dot(self.w,self.x[i,:])+self.b))<=0:
                    # print('有错',i)
                    return i
            i += 1
            if i==self.row:
                # print("无错",i)
                return -1
    '''
    更新weight:计算nyx
    '''
    def update_w(self,num):
        # print("weight的增值",self.l*self.x[num,:]*self.y[num])
        self.w=self.w+self.l*self.x[num,:]*self.y[num]
        return
    def update_b(self,num):
        # print("bias的增值",self.l*self.y[num])
        self.b=self.b+self.l*self.y[num]
        return

def main():
    lr=1
    xtrain=np.array([[3,3],[2,2],[1,1]])
    ytrain=np.array([1,1,-1])
    p=Perceptron(xtrain=xtrain,ytrain=ytrain,lr=lr)
    '''
    循环直到没有误分类点
    '''
    i=0
    while True:
        i+=1
        num=p.mis_dot()
        if num==-1:
            weight=p.w
            bias=p.b
            print(weight,bias,"终止")
            break
        else:
            p.update_w(num)
            p.update_b(num)
    return

if __name__=="__main__":
    main()
