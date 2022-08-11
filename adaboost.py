'''
    自编程实现决策树的AdaBoost
'''
import numpy as np
def loadData():
    '''
    读入数据函数
    :return:
        dataSet:数据矩阵
        classLabels:分类标签
    '''
    dataSet=np.matrix([
        [0., 1., 3.],
        [0., 3., 1.],
        [1., 2., 2.],
        [1., 1., 3.],
        [1., 2., 3.],
        [0., 1., 2.],
        [1., 1., 2.],
        [1., 1., 1.],
        [1., 3., 1.],
        [0., 2., 1.]

    ])
    classLabels=np.matrix([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    return dataSet,classLabels

def stumpClassify(dataSet,threshVal,threshBiaozhi,dim):
    '''
    单层决策树分类函数
    Paremeters:
        dataSet:待分类的数据矩阵
        threshVal:阈值
        threshBiaozhi:分类标志，大于或小于
        dim:维数，即特征
    :return
        classArr:分类结果矩阵
    '''
    classArr=np.ones((np.shape(dataSet)[0],1))          # 初始化分类矩阵为全一
    if threshBiaozhi=='ld':
        classArr[dataSet[:,dim]<=threshVal]=-1          # 将dataSet第dim维度特征值小于等于阈值的分类为负
    else:
        classArr[dataSet[:,dim]>threshVal]=-1
    return classArr

def buildStump(dataSet,classLabels,D):
    '''
    Parameters:
        D:权值矩阵
        err:损失误差
        stepSize:步长
        threshVal:阈值
    :return:
        bestStump:最佳决策树信息
        bestClass:最佳分类结果
        minErr:最小误差
    '''
    dataSet=np.mat(dataSet)
    classLabels=np.mat(classLabels).T
    bestStump={}
    m,n=np.shape(dataSet)
    bestClass=np.mat(np.zeros((m,1)))
    minErr=float('inf')
    stepNum=int(10)
    for i in range(n):
        Min=dataSet[:,i].min()
        Max=dataSet[:,i].max()
        stepSize=(Max-Min)/stepNum
        for j in range(-1,stepNum+1):
            for biaozhi in ['ld','gt']:
                threshVal = Min + stepSize * j
                err = np.mat(np.ones((m, 1)))
                predictClass=stumpClassify(dataSet,threshVal,biaozhi,i)
                err[predictClass==classLabels]=0
                err=D.T*err
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                #     i, threshVal, biaozhi, err))
                if err<minErr:
                    minErr=err
                    bestClass=predictClass
                    bestStump['thresh']=threshVal
                    bestStump['biaozhi']=biaozhi
                    bestStump['dim']=i
    return bestClass,bestStump,minErr

def train(dataSet,classLabels,numIt=40):
    '''
    AdaBoost决策树训练函数
    Paremeters:
        numIt:迭代次数
    :return:
        weekClassArr:完整决策树信息
        aggClassEst:接受权值分布
    '''
    m,n=np.shape(dataSet)
    D=np.mat(np.ones((m,1))/m)  # 权值初始化
    # 初始化：弱分类器桩
    weekClassArr=[]
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestClass, bestStump, minErr=buildStump(dataSet,classLabels,D)
        alpha=float(0.5*np.log((1-minErr)/max(minErr, 1e-16)))      # 计算系数alpha
        bestStump['alpha']=alpha
        weekClassArr.append(bestStump)
        epsilon=np.multiply(-1*alpha*bestClass,classLabels.T)
        # 权值更新
        D=np.multiply(D,np.exp(epsilon))
        D=D/D.sum()
        # 构建基本分类器的线性组合
        aggClassEst+=alpha*bestClass
        # 计算是否全部分类正确
        aggError=np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        errRate=aggError.sum()/m
        if errRate==0:
            break
    return weekClassArr,aggClassEst


if __name__=='__main__':
    dataSet,classLabels=loadData()
    weekClassArr, aggClassEst=train(dataSet,classLabels,numIt=40)
    print('最终的分类器是：\n',weekClassArr)
    print('最终权值为：\n',aggClassEst)






