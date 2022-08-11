'''
    使用sklearn库
    CART算法（剪枝）
'''
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
def main():
    start=time.time()

    # 数据读取
    features=['age','job','house','credit']
    X_dataSet=[
        ["青年", "否", "否", "一般"],
        ["青年", "否", "否", "好"],
        ["青年", "是", "否", "好"],
        ["青年", "是", "是", "一般"],
        ["青年", "否", "否", "一般"],
        ["中年", "否", "否", "一般"],
        ["中年", "否", "否", "好"],
        ["中年", "是", "是", "好"],
        ["中年", "否", "是", "非常好"],
        ["中年", "否", "是", "非常好"],
        ["老年", "否", "是", "非常好"],
        ["老年", "否", "是", "好"],
        ["老年", "是", "否", "好"],
        ["老年", "是", "否", "非常好"],
        ["老年", "否", "否", "一般"]
    ]
    x_train=pd.DataFrame(X_dataSet,columns=features)
    print(x_train)
    y_train=pd.DataFrame(["否", "否", "是", "是", "否", "否", "否", "是", "是", "是", "是", "是", "是", "是", "否"])

    # 数据预处理
    le_x=preprocessing.LabelEncoder()
    le_x.fit(np.unique(x_train))
    # 此处注意transform只能应用于一维，所有使用lamda函数进行函数式编程
    x_train=x_train.apply(le_x.transform)
    print('预处理x:\n',x_train)
    le_y=preprocessing.LabelEncoder()
    le_y.fit(np.unique(y_train))
    y_train=le_y.transform(y_train)
    print('预处理y:\n',y_train)

    # 模型创建和训练
    clf=DecisionTreeClassifier(criterion='gini')
    clf.fit(x_train,y_train)

    # 模型预测
    record=pd.DataFrame([['青年','否','是','一般']])
    record=record.apply(le_x.transform)
    predict=clf.predict(record)

    # 结果输出
    X_show = [{features[i]: record.values[0][i]} for i in range(len(features))]
    print('测试结果:')
    print("{0}被分类为:{1}".format(X_show, le_y.inverse_transform(predict)))
    print("time:{:.4f}s".format(time.time() - start))


if __name__=='__main__':
    main()



