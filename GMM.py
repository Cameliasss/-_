'''
    自编程实现GMM算法
    使用EM算法实现高斯混合模型
    题目9.3，参照算法9.2
'''
import numpy as np
def loadData():
    '''
    :return:
        Y:观测数据
    '''
    Y= np.array([-67,-48,6,8,14,16,23,24,28,29,41,49,56,60,75]).reshape(1,15)
    return Y

def gaussianDistribution(dataSet,mean,sigma):
    '''
    计算高斯分布密度函数
    Parameter:
        mean:均值
        sigma：方差
        dataSet:观测数据集
    :return:
        result:高斯分布密度
    '''
    return 1/np.sqrt(2*np.pi*sigma)*np.exp(-1*((dataSet-mean)**2)/(2*sigma))

def update_z(dataSet,mean,sigma,alpha):
    '''
    更新响应度函数
    '''
    xiangying=alpha*gaussianDistribution(dataSet,mean,sigma)
    # axis=0是每一列相加，行数不变；反之，axis=1是每一行相加
    xiangying=xiangying/xiangying.sum(axis=0)
    return xiangying

def update(dataSet,mean,sigma,alpha,k):
    '''
    更新参数函数
    '''
    xiangying=update_z(dataSet,mean,sigma,alpha)
    mean_new=((xiangying*dataSet).sum(axis=1)/xiangying.sum(axis=1)).reshape(k,1)
    sigma_new=((xiangying*(dataSet-mean)**2).sum(axis=1)/xiangying.sum(axis=1)).reshape(k,1)
    alpha_new=(xiangying.sum(axis=1)/np.size(dataSet)).reshape(k,1)
    return mean_new,sigma_new,alpha_new

def judgeStop(mean,mean_old,sigma,sigma_old,alpha,alpha_old,tol):
    '''
    判断迭代终止函数
    :parameter
        tol:阈值
    '''
    a=np.linalg.norm(mean-mean_old)
    b=np.linalg.norm(sigma-sigma_old)
    c=np.linalg.norm(alpha-alpha_old)
    print('损失误差为{}'.format(np.sqrt(a**2+b**2+c**2)))
    return True if np.sqrt(a**2+b**2+c**2)<tol else False

def train(iter=500):
    '''
    EM训练函数
    :parameter:
        iter:迭代次数
    '''
    y = loadData()
    k=2
    alpha=np.array([1/k for i in range(k)],dtype='float16').reshape(k,1)
    mean=np.array([y.mean() for i in range(k)],dtype='float16').reshape(k,1)
    sigma=np.array([np.std(y)**2 for i in range(k)],dtype='float16').reshape(k,1)
    step=0
    while step<iter:
        step+=1
        print('\nstep=',step)
        print('参数值为：alpha={},mean={},sigam={}'.format(alpha.reshape(1,k),mean.reshape(1,k),sigma.reshape(1,k)))
        alpha_old=alpha.copy()
        mean_old=mean.copy()
        sigma_old=sigma.copy()
        mean,sigma,alpha=update(y,mean,sigma,alpha,k)
        if judgeStop(mean,mean_old,sigma,sigma_old,alpha,alpha_old,tol=1e-15):
            print('\n\n训练结束！\n参数值为：alpha={},mean={},sigma={}'.format(alpha.reshape(1,k),mean.reshape(1,k),sigma.reshape(1,k)))
            break
    return alpha, mean, sigma

if __name__=='__main__':
    train()


