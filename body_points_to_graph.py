# 根据人体关键点检测出的关键点，构建图
from body_from_image_fc import *
import dgl
import numpy as np
import networkx as nx
import torch

# 检测人体关键点
body_points = body_points("C:/Users/lab/Desktop/taskey/action_video_test/2.jpg")

body_graph = [] # 空列表用于存储graph

# graph中src和dst节点(人体关键点指向关系)
src = np.array([17, 15, 18, 16, 0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 10, 13, 11, 14, 23, 22, 24, 20, 19, 21])
dst = np.array([15, 0, 16, 0, 1, 1, 2, 3, 1, 5, 6, 1, 8, 8, 9, 12, 10, 13, 22, 11, 11, 19, 14, 14])

for points in body_points:
    g = dgl.DGLGraph((src, dst)) # 建立graph
    # nx.draw(g.to_networkx(), with_labels=True)
    node_feature = points
    g.ndata['coordinate'] = torch.tensor(node_feature)
    body_graph.append(g)
    
    # graph可视化
    pos = node_feature[:,0:2] # 取前两列坐标（第三列为置信度），作为显示时的节点坐标
    pos[:, 1] = -pos[:, 1]  # 由于默认显示坐标原点再左上角，鉴于人眼观察习惯，将其坐标上下翻转
    nx.draw(g.to_networkx(), pos = pos, with_labels=True) # 显示graph
    


