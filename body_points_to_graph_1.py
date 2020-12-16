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

os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

torch.manual_seed(100)

# 导入实验小型数据集（站立&坐下）
img_filenames_sit = 'C:\\Users\\lab\\Desktop\\taskey\\action_video_test\\sit'
img_filenames_stand = 'C:\\Users\\lab\\Desktop\\taskey\\action_video_test\\stand'
files_1 = os.listdir(img_filenames_sit)
files_2 = os.listdir(img_filenames_stand)

# graph中src和dst节点(人体关键点指向关系)
src = np.array([17, 15, 18, 16, 0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 10, 13, 11, 14, 23, 22, 24, 20, 19, 21])
dst = np.array([15, 0, 16, 0, 1, 1, 2, 3, 1, 5, 6, 1, 8, 8, 9, 12, 10, 13, 22, 11, 11, 19, 14, 14])

# 用于根据图片检测人体关键点
def detect_body_point_and_save(img_filenames, imgs, save_name, label):
    # 若已存在对应的.npy，则直接导入，不再重新检测关键点
    if os.path.exists(save_name):
        print('Body Point data have already existed')
        bodies_points = np.load(save_name)
        
    else:
        # 若无对应.npy，则基于openpose进行人体关键点检测
        bodies_points = []
        for img in imgs:
            # print(img_filenames + img)
            body_point = get_body_points(img_filenames + img)
            bodies_points.append(body_point)
            bodies_points_array = np.array(bodies_points)
            np.save(save_name, bodies_points_array)
        
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
            body_graph.append([g, torch.tensor([label]).float()])
    return body_graph

# 分别生成坐姿与站姿的graph及其label
body_graph_sit = detect_body_point_and_save(img_filenames_sit, files_1, 'sit.npy', (0,1))
body_graph_stand = detect_body_point_and_save(img_filenames_stand, files_2, 'stand.npy', (1,0))
# 合并为一个list
body_graph = body_graph_sit + body_graph_stand

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

# 搭建GCN模型
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
            # output = F.softmax(self.fc(read_out))
            output = self.fc(read_out)
            return output

# graph中src和dst节点(人体关键点指向关系)
src = np.array([17, 15, 18, 16, 0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 10, 13, 11, 14, 23, 22, 24, 20, 19, 21])
dst = np.array([15, 0, 16, 0, 1, 1, 2, 3, 1, 5, 6, 1, 8, 8, 9, 12, 10, 13, 22, 11, 11, 19, 14, 14])
graph = dgl.DGLGraph((src, dst)) # 建立graph
graph = dgl.add_self_loop(graph)
 
model = GCN(3, 20, 2)  

# 只训练一张图，检查模型能否正常运行
inputs = [body_graph_sit[0][0], body_graph_stand[1][0]]
label = [body_graph_sit[0][1], body_graph_stand[1][1]]
              
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
for epoch in range(50):
    for i in range(2):
        output = model(graph, inputs[i].ndata['coordinate'].float())
        pred = torch.argmax(output, axis=1)
        loss = nn.BCEWithLogitsLoss()
        loss = loss(output, label[i])
        # loss.requires_grad = True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        
        if (np.array(label[i])[0] == [0., 1.]).all():
            label_ = '坐姿'
        else:
            label_ = '站姿'
            
        if (pred == torch.tensor([i])).all():
            pred_ = '坐姿'
        else:
            pred_ = '站姿'
        print('Epoch: {}; Label: {}'.format(epoch, label_))
        # print('Epoch: {}; Output: {}'.format(epoch, output))
        print('Epoch: {}; Predict: {}'.format(epoch, pred_))
        print('Epoch: {}; Loss: {}'.format(epoch, np.array(loss.item())))
        print('-------------------------------------------------')