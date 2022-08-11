'''
    C4.5生成决策树
'''
from math import log

# 数据读入
def loadData():
    dataSet=[
        ['青年', '否', '否', '一般', '否'],
        ['青年', '否', '否', '好', '否'],
        ['青年', '是', '否', '好', '是'],
        ['青年', '是', '是', '一般', '是'],
        ['青年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '好', '否'],
        ['中年', '是', '是', '好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '好', '是'],
        ['老年', '是', '否', '好', '是'],
        ['老年', '是', '否', '非常好', '是'],
        ['老年', '否', '否', '一般', '否']
    ]
    labels = ['age', 'job', 'house', 'credit','class']
    return dataSet,labels

# 计算信息增益比
# 经验熵
def cal_entropy(dataSet,index=-1):
    label_count={}
    for data in dataSet:
        label=data[index]
        if label not in label_count:
            label_count[label]=0
        label_count[label]+=1
    entropy=-sum([(p/len(dataSet))*log(p/len(dataSet),2) for p in label_count.values()])
    print('经验熵:',entropy)
    return entropy

# 经验条件熵
def cal_conditional_entropy(dataSet,index=0):
    feature_data={}
    for data in dataSet:
        feature=data[index]
        if feature not in feature_data:
            feature_data[feature]=[]
        feature_data[feature].append(data)
    conditionalEntropy=sum([(len(p)/len(dataSet))*cal_entropy(p) for p in feature_data.values()])
    print('经验条件熵:',conditionalEntropy)
    return conditionalEntropy

# 信息增益
def info_gain(entropy,conditionalEntropy):
    return entropy-conditionalEntropy

# 信息增益比
def info_gain_ratio(entropy,infoGain):
    if infoGain==0:
        return 0
    else:
        return infoGain/entropy

# 决策树生成
# 计算最优特征
def find_best_feature(dataSet,labels):
    _entropy=cal_entropy(dataSet)
    features=[]
    for index in range(len(dataSet[0])-1):
        _conditionalEntropy=cal_conditional_entropy(dataSet,index)
        _info_gain=info_gain(_entropy,_conditionalEntropy)
        _info_gain_ratio=info_gain_ratio(_entropy,_info_gain)
        features.append((index,_info_gain_ratio))
        print('特征({})的信息增益比是{:.3f}'.format(labels[index],_info_gain_ratio))
    best_feature=max(features,key=lambda x:x[-1])
    print('最优特征是{},信息增益比是{:.3f}'.format(labels[best_feature[0]],best_feature[-1]),'\n')
    return best_feature

# C4.5算法
def info_gain_train(dataSet,labels):
    # 是否是单类别判断
    label_count={}
    for data in dataSet:
        label=data[-1]
        if label not in label_count:
            label_count[label]=0
        label_count[label]+=1
    if len(label_count.keys())==1:
        key=list(label_count.keys())[0]
        print('单类别为{}'.format(key))
        return
    # 分支
    # 计算不同特征值的统计值
    best_feature=find_best_feature(dataSet,labels)
    feature_data={}
    for data in dataSet:
        feature=data[best_feature[0]]
        if feature not in feature_data:
            feature_data[feature]=[]
        feature_data[feature].append(data)
    # 选出对应特征类别
    for data in zip(feature_data.keys(),feature_data.values()):
        print('当前节点特征{}的取值为\'{}\''.format(labels[best_feature[0]],data[0]))
        info_gain_train(data[1],labels)
    return

if __name__=='__main__':
    dataSet,labels=loadData()
    info_gain_train(dataSet,labels)
