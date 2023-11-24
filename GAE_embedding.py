import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output


class GCN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, embedding_dim):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, embedding_dim)

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = torch.relu(x)
        x = self.gcn2(x, adj)
        return x

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, original_adj, feature_matrix):
        normalized_feature_matrix = torch.nn.functional.normalize(feature_matrix, p=2, dim=1)
        new_adj = torch.mm(normalized_feature_matrix, normalized_feature_matrix.t())
        return self.mse_loss(original_adj, new_adj)

class newCustomLoss(nn.Module):
    def __init__(self):
        super(newCustomLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
    def forward(self, originalMatrix, featureMatrix):
        normalized_feature_matrix = torch.nn.functional.normalize(featureMatrix, p=2, dim=1)
        new_adj = torch.mm(normalized_feature_matrix, normalized_feature_matrix.t())
        originalMatrixX = originalMatrix.shape(0)
        originalMatrixY = originalMatrix.shape(1)
        mseListTempx = []
        mseListTempy = []
        originalList = []
        newList = []
        for i in range(0, originalMatrixX):
            for j in range(0, originalMatrixY):
                if originalMatrix[i][j] != 0:
                    mseListTempx.append(i)
                    mseListTempy.append(j)


                    originalList.append(originalMatrix[i][j])
                    newList.append(new_adj[i][j])
        return self.mse_loss(originalList, newList)


# 生成drug邻接矩阵
def generate_fused_drug_similarity_data(dataset_directory):
    D_SSM1 = np.loadtxt(dataset_directory + '/drugsim1network.txt')
    D_SSM2 = np.loadtxt(dataset_directory + '/drugsim2network.txt')
    D_SSM3 = np.loadtxt(dataset_directory + '/drugsim3network.txt')
    D_SSM4 = np.loadtxt(dataset_directory + '/drugsim4network.txt')
    D_SSM5 = np.loadtxt(dataset_directory + '/drugsim5network.txt')
    D_SSM6 = np.loadtxt(dataset_directory + '/drugsim6network.txt')
    D_SSM = (D_SSM1 + D_SSM2 + D_SSM3 + D_SSM4 + D_SSM5 + D_SSM6) / 6
    # 为了把里面的自连边删除，需要把自连边的相似度修改为0
    for i in range(0, len(D_SSM)):
        D_SSM[i][i] = 0
    return D_SSM


# 生成target邻接矩阵
def generate_fused_target_similarity_data(dataset_directory):
    T_SSM1 = np.loadtxt(dataset_directory + '/proteinsim1network.txt')
    T_SSM2 = np.loadtxt(dataset_directory + '/proteinsim2network.txt')
    T_SSM3 = np.loadtxt(dataset_directory + '/proteinsim3network.txt')
    T_SSM4 = np.loadtxt(dataset_directory + '/proteinsim4network.txt')
    T_SSM = (T_SSM1 + T_SSM2 + T_SSM3 + T_SSM4) / 4
    # 为了把里面的自连边删除，需要把自连边的相似度修改为0
    for i in range(0, len(T_SSM)):
        T_SSM[i][i] = 0
    return T_SSM


drug_similarity_data = generate_fused_drug_similarity_data("data")
target_similarity_data = generate_fused_target_similarity_data("data")
drug_similarity_data_torch = torch.tensor(drug_similarity_data, dtype=torch.float)
target_similarity_data_torch = torch.tensor(target_similarity_data, dtype=torch.float)

number_of_drug_nodes = drug_similarity_data.shape[0]
number_of_target_nodes = target_similarity_data.shape[0]
print("药物有%d个" % (number_of_drug_nodes))
print("靶点有%d个" % (number_of_target_nodes))

input_dim = 128
hidden_dim = 128
embedding_dim = 128
epochs = 1000000
lambda_l2 = 0.01  # L2正则化系数
custom_loss = CustomLoss()

drug_gcn = GCN(number_of_drug_nodes, input_dim, hidden_dim, embedding_dim)
drug_optimizer = optim.Adam(drug_gcn.parameters(), lr=0.000001)
target_gcn = GCN(number_of_target_nodes, input_dim, hidden_dim, embedding_dim)
target_optimizer = optim.Adam(target_gcn.parameters(), lr=0.000001)

# 生成一个形状为(num_nodes, input_dim)的随机初始节点特征矩阵
drug_x = torch.randn(number_of_drug_nodes, input_dim)
drug_x = drug_x.to(device)
drug_gcn = drug_gcn.to(device)
drug_similarity_data_torch = drug_similarity_data_torch.to(device)
drug_mse_loss_values = []
drug_l2_loss_values = []
drug_total_loss_values = []

# 生成一个形状为(num_nodes, input_dim)的随机初始节点特征矩阵
target_x = torch.randn(number_of_target_nodes, input_dim)
target_x = target_x.to(device)
target_gcn = target_gcn.to(device)
target_similarity_data_torch = target_similarity_data_torch.to(device)
target_mse_loss_values = []
target_l2_loss_values = []
target_total_loss_values = []

for epoch in range(epochs):
    drug_optimizer.zero_grad()
    drug_z = drug_gcn(drug_x, drug_similarity_data_torch)

    # 计算均方差误差
    drug_mse_loss = custom_loss(drug_similarity_data_torch, drug_z)
    drug_mse_loss_values.append(float(drug_mse_loss))
    # 计算L2正则化
    drug_l2_reg = torch.tensor(0.0).to(device)
    for param in drug_gcn.parameters():
        drug_l2_reg = drug_l2_reg + torch.norm(param, p=2).to(device)
    drug_l2_loss_values.append(float(lambda_l2 * drug_l2_reg))

    # 计算总损失
    drug_loss = drug_mse_loss + lambda_l2 * drug_l2_reg
    drug_total_loss_values.append(float(drug_loss))
    drug_loss.backward()
    drug_optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(
            f'Epoch: {epoch + 1}, MSE Loss: {drug_mse_loss.item()}, L2 Loss: {lambda_l2 * drug_l2_reg}, Total Loss: {drug_loss.item()}')

for epoch in range(epochs):
    target_optimizer.zero_grad()
    target_z = target_gcn(target_x, target_similarity_data_torch)

    # 计算均方差误差
    target_mse_loss = custom_loss(target_similarity_data_torch, target_z)
    target_mse_loss_values.append(float(target_mse_loss))
    # 计算L2正则化
    target_l2_reg = torch.tensor(0.0).to(device)
    for param in target_gcn.parameters():
        target_l2_reg = target_l2_reg + torch.norm(param, p=2).to(device)
    target_l2_loss_values.append(float(lambda_l2 * target_l2_reg))

    # 计算总损失
    target_loss = target_mse_loss + lambda_l2 * target_l2_reg
    target_total_loss_values.append(float(target_loss))
    target_loss.backward()
    target_optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(
            f'Epoch: {epoch + 1}, MSE Loss: {target_mse_loss.item()}, L2 Loss: {lambda_l2 * target_l2_reg}, Total Loss: {target_loss.item()}')

# 创建一个新的图形
plt.figure()
plt.plot(drug_mse_loss_values)
plt.title("MSE Loss Curve for drug embeddings")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("output/MSE Loss Curve for drug embeddings.png")

# 创建一个新的图形
plt.figure()
plt.plot(drug_l2_loss_values)
plt.title("L2 Loss Curve for drug embeddings")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("output/L2 Loss Curve for drug embeddings.png")

# 创建一个新的图形
plt.figure()
plt.plot(drug_total_loss_values)
plt.title("Total Loss Curve for drug embeddings")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("output/Total Loss Curve for drug embeddings.png")

# 创建一个新的图形
plt.figure()
plt.plot(target_mse_loss_values)
plt.title("MSE Loss Curve for target embeddings")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("output/MSE Loss Curve for target embeddings.png")

# 创建一个新的图形
plt.figure()
plt.plot(target_l2_loss_values)
plt.title("L2 Loss Curve for target embeddings")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("output/L2 Loss Curve for target embeddings.png")

# 创建一个新的图形
plt.figure()
plt.plot(target_total_loss_values)
plt.title("Total Loss Curve for target embeddings")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("output/Total Loss Curve for target embeddings.png")

with torch.no_grad():
    drug_z = drug_gcn(drug_x, drug_similarity_data_torch)
    target_z = target_gcn(target_x, target_similarity_data_torch)

drug_z_pandas = pd.DataFrame(drug_z.cpu().detach().numpy())
target_z_pandas = pd.DataFrame(target_z.cpu().detach().numpy())
GAE_total_pandas = pd.concat([drug_z_pandas,target_z_pandas],ignore_index=True)
GAE_total_pandas = GAE_total_pandas.reset_index()
GAE_total_pandas = GAE_total_pandas.rename(columns={'index':'node_id'})

GAE_total_pandas.to_csv("output/GAE_total_pandas.csv")
