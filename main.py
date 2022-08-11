'''
    sklearn实现
'''
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

def main():
    dataSet=np.array([
        [0, 1, 3],
        [0, 3, 1],
        [1, 2, 2],
        [1, 1, 3],
        [1, 2, 3],
        [0, 1, 2],
        [1, 1, 2],
        [1, 1, 1],
        [1, 3, 1],
        [0, 2, 1]

    ])
    Y=np.array([-1,-1,-1,-1,-1,-1,1,1,-1,-1])
    clf=AdaBoostClassifier(learning_rate=0.5)
    clf.fit(dataSet,Y)
    y_predict=clf.predict(dataSet)
    score=clf.score(dataSet,Y)
    print('原始输出：{}'.format(Y))
    print('原始预测值：{}'.format(y_predict))
    print('预测的正确率：{:2f}'.format(score))
    return

if __name__=='__main__':
    main()


