import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def main():
    # 训练数据
    X_train=np.array([[5,4],
                      [9,6],
                      [4,7],
                      [2,3],
                      [8,1],
                      [7,2]])
    y_train=np.array([1,1,1,-1,-1,-1])
    # 待预测数据
    X_new = np.array([[5, 3]])
    # 不同k值对结果的影响
    for k in range(1,6):
        # 创建分类器对象
        clf = KNeighborsClassifier(n_neighbors=k)
        # 用训练器数据拟合模型
        clf.fit(X_train, y_train)
        # 预测
        y_predict=clf.predict(X_new)
        print("k={},被分类为：{}".format(k,y_predict))

if __name__=="__main__":
    main()
