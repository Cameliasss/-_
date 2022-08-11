'''
    sklearn模块学习支持向量机
'''
import numpy as np
from sklearn import svm

def main():
    x_train=np.array([[1,2],[2,3],[3,3],[2,1],[3,2]])
    y_train=np.array([1,1,1,-1,-1])
    x_new=np.array([[4,1]])
    methods=['linear', 'poly', 'rbf', 'sigmoid']
    for method in methods:
        clf=svm.SVC(kernel=method)
        clf.fit(x_train,y_train)
        y_predict=clf.predict(x_new)
        print('输入{}的类别是{}'.format(x_new,y_predict))
        print('方法{}的参数为：{}'.format(method,clf.support_vectors_))
    return

if __name__=="__main__":
    main()
