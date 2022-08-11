'''
    sklearn模块实现
    测试随机下降、牛顿、拟牛顿法
'''
import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    x_train=np.array([[3,3,3],[4,3,2],[2,1,2],[1,1,1]])
    y_train=np.array([1,1,1,0])
    x_new=np.array([[1, 2, -2]])
    x_test = np.array([[-1,0,1],[2,-2,1]])
    y_test = np.array([0,0])
    methods=['liblinear','lbfgs','newton-cg','sag','saga']
    res=[]
    for method in methods:
        clf=LogisticRegression(solver=method,max_iter=100,intercept_scaling=2)
        clf.fit(x_train,y_train)
        y_predict=clf.predict(x_new)
        correct_rate=clf.score(x_test,y_test)
        res.append((method,correct_rate))
        print('{}方法的正确率是{}'.format(res[-1][0],res[-1][1]))
        print('输入{}的预测值为{}'.format(x_new,y_predict))
    return

if __name__=='__main__':
    main()