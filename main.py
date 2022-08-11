import math
'''
用kd树计算最邻近点算法
构建平衡kd树：1、指标：深度与维数的关系
            2、方式：求中位数--求对应维数中位数函数
            3、采用递归
'''
class kdTree:
    def __init__(self,depth=0):
        self.key=None
        self.left=None
        self.right=None
        self.depth=depth
    # 递归创建kd树
    def insert_node(self,data,depth):
        if not data:
            return
        dims=len(data[0])
        axis=depth%dims
        data.sort(key=lambda x:x[axis])
        median_index=len(data)//2
        self.key=data[median_index]
        self.left=kdTree(depth=depth+1)
        self.right=kdTree(depth=depth+1)
        self.left.insert_node(data[:median_index],depth=depth+1)
        self.right.insert_node(data[median_index+1:],depth=depth+1)
    # 前序遍历
    def pre_order(self):
        if not self.key:
            return
        print(self.key,self.depth)
        self.left.pre_order()
        self.right.pre_order()
    # 查询叶结点
    def query(self,data,value,depth):
        dims=len(data[0])
        axis=depth%dims
        DEPTH=math.floor(math.log2(len(data)))
        if not self.key:
            return
        if self.key==value:
            return self.key
        if depth==DEPTH:
            return self.key
        if value[axis]<self.key[axis]:
            if self.left.key==None:
                return self.key
            return self.left.query(data,value,depth+1)
        else:
            if self.right.key==None:
                return self.key
            return self.right.query(data,value,depth+1)
    # 查询第一个最近点
    def query_neast(self,value,leaf,dims,depth):
        axis = depth % dims
        if self.key==leaf:
            distance_nest = cal_distance(value, leaf, dims)
            distance_root=cal_distance(value,self.key,dims)
            if distance_root<=distance_nest:
                distance_nest=distance_root
                leaf=self.key
            return leaf
        if self.left.key==leaf:
            distance_nest=cal_distance(value,leaf,dims)
            distance_xd=cal_distance(value,self.right.key,dims)
            distance_fuqin=cal_distance(value,self.key,dims)
            if distance_fuqin<=distance_nest:
                distance_nest=distance_fuqin
                leaf=self.key
            if distance_xd<=distance_nest:
                leaf=self.right.key
            return leaf
        if self.right.key==leaf:
            distance_nest=cal_distance(value,leaf,dims)
            distance_xd=cal_distance(value,self.left.key,dims)
            distance_fuqin = cal_distance(value, self.key, dims)
            if distance_fuqin <= distance_nest:
                distance_nest = distance_fuqin
                leaf = self.key
            if distance_xd<=distance_nest:
                leaf=self.left.key
            return leaf
        if leaf[axis]<self.key[axis]:
            return self.left.query_neast(value,leaf,dims,depth+1)
        else:
            return self.right.query_neast(value,leaf,dims,depth+1)


# 求两点间距离
def cal_distance(x1,x2,dims):
    sum=0
    for i in range(dims):
        sum+=math.pow(x1[i]-x2[i],2)
    return math.sqrt(sum)

if __name__=="__main__":
    dataSet = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    dims=len(dataSet[0])
    DEPTH = math.floor(math.log2(len(dataSet)))
    value=[3,4.5]
    kd_tree=kdTree(depth=0)
    kd_tree.insert_node(dataSet,depth=0)
    '''
    kd树搜索：1、找到目标点x对应的叶子节点
            2、找父节点
            3、找堂兄弟
            4、递归回退
    '''
    leaf=kd_tree.query_neast(value, list(kd_tree.query(dataSet,value,0)), dims, 0)
    for i in range(DEPTH-1):
        leaf=kd_tree.query_neast(value,leaf,dims,0)
    print("最临近点是：",leaf)