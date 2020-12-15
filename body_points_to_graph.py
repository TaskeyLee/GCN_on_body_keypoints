# 根据人体关键点检测出的关键点，构建图
from body_from_image_fc import get_body_points
import dgl
import numpy as np
import networkx as nx
import torch
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import cv2
import matplotlib.pyplot as plt

torch.manual_seed(100)

# 导入实验小型数据集（站立&坐下）
files_1 = os.listdir('C:/Users/lab/Desktop/taskey/action_video_test/sit')
files_2 = os.listdir('C:/Users/lab/Desktop/taskey/action_video_test/stand')

# graph中src和dst节点(人体关键点指向关系)
src = np.array([17, 15, 18, 16, 0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 10, 13, 11, 14, 23, 22, 24, 20, 19, 21])
dst = np.array([15, 0, 16, 0, 1, 1, 2, 3, 1, 5, 6, 1, 8, 8, 9, 12, 10, 13, 22, 11, 11, 19, 14, 14])

# 检测人体关键点
bodies_points = []
for img in files_1:
    body_point = get_body_points('C:/Users/lab/Desktop/taskey/action_video_test/sit/' + img)
    bodies_points.append(body_point)
    
body_graph = [] # 空列表用于存储graph


for body_points in bodies_points: # 遍历所有检测图片结果
    for body_point in body_points: # 由于某些图片中存在多个人，所以再次再同一张图片结果中遍历所有人体
        g = dgl.DGLGraph((src, dst)) # 建立graph
        g = dgl.add_self_loop(g)
        plt.figure()
        pos = body_point[:,0:2] # 取前两列坐标（第三列为置信度），作为显示时的节点坐标
        pos[:, 1] = -pos[:, 1]  # 由于默认显示坐标原点再左上角，鉴于人眼观察习惯，将其坐标上下翻转
        nx.draw(g.to_networkx(), pos = pos, with_labels=True) # 显示graph
        node_feature = body_point
        g.ndata['coordinate'] = torch.tensor(node_feature)
        body_graph.append([g, torch.tensor(0)])

bodies_points = []
for img in files_2:
    body_point = get_body_points('C:/Users/lab/Desktop/taskey/action_video_test/stand/' + img)
    bodies_points.append(body_point)
    
for body_points in bodies_points: # 遍历所有检测图片结果
    for body_point in body_points: # 由于某些图片中存在多个人，所以再次再同一张图片结果中遍历所有人体
        g = dgl.DGLGraph((src, dst)) # 建立graph
        g = dgl.add_self_loop(g)
        plt.figure()
        pos = body_point[:,0:2] # 取前两列坐标（第三列为置信度），作为显示时的节点坐标
        pos[:, 1] = -pos[:, 1]  # 由于默认显示坐标原点再左上角，鉴于人眼观察习惯，将其坐标上下翻转
        nx.draw(g.to_networkx(), pos = pos, with_labels=True) # 显示graph
        node_feature = body_point / 100
        g.ndata['coordinate'] = torch.tensor(node_feature)
        body_graph.append([g, torch.tensor(1)])

# 构建pytorch所使用的dataloader

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels

dataloader = DataLoader(
    body_graph,
    collate_fn=collate,
    shuffle = True)
        
    
#         # graph可视化
#         pos = node_feature[:,0:2] # 取前两列坐标（第三列为置信度），作为显示时的节点坐标
#         pos[:, 1] = -pos[:, 1]  # 由于默认显示坐标原点再左上角，鉴于人眼观察习惯，将其坐标上下翻转
#         nx.draw(g.to_networkx(), pos = pos, with_labels=True) # 显示graph

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_class):
        super(GCN, self).__init__()
        self.GConv1 = dglnn.GraphConv(in_dim, hidden_dim) # in_dim指的是每个节点特征的维度，而不是节点数，所有图结构本身特点均由输入的graph决定
        self.GConv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_class)
        
    def forward(self, g, inputs):
        h = F.relu(self.GConv1(g, inputs)) # 图卷积一次，激活一次
        h = F.relu(self.GConv2(g, h)) # 图卷积两次，激活一次
        with g.local_scope():
            g.ndata['features'] = h
            read_out = dgl.mean_nodes(g, 'features')
            output = F.log_softmax(self.fc(read_out))
            return output

# graph中src和dst节点(人体关键点指向关系)
src = np.array([17, 15, 18, 16, 0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 10, 13, 11, 14, 23, 22, 24, 20, 19, 21])
dst = np.array([15, 0, 16, 0, 1, 1, 2, 3, 1, 5, 6, 1, 8, 8, 9, 12, 10, 13, 22, 11, 11, 19, 14, 14])
graph = dgl.DGLGraph((src, dst)) # 建立graph
graph = dgl.add_self_loop(g)
 
model = GCN(3, 10, 2)                
optimizer = torch.optim.Adam(model.parameters(), lr = 0.000000001)
for epoch in range(10):
    for idx, (features, labels) in enumerate(dataloader):
        # feats = body_graph[0].ndata['coordinate'].float()
        output = model(graph, features.ndata['coordinate'].float())
        loss = F.nll_loss(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if idx % 10 == 0:
            print('Epoch: {}; Output: {}'.format(epoch, np.array(output.data)))
            print('Epoch: {}; Loss: {}'.format(epoch, np.array(loss.item())))