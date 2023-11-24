import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")

heads = 8

# 设置GAT
class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels1, hidden_channels2, hidden_channels3):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATv2Conv(num_features, hidden_channels1, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels1*heads, hidden_channels2, heads=heads)
        self.conv3 = GATConv(hidden_channels2*heads, hidden_channels3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.conv1(x, edge_index)
        x_relu1 = F.relu(x1)
        x_dropout1 = F.dropout(x_relu1, p=0.5, training=self.training)

        x2 = self.conv2(x_dropout1, edge_index)
        x_relu2 = F.relu(x2)
        x_dropout2 = F.dropout(x_relu2, p=0.5, training=self.training)

        x3 = self.conv3(x_dropout2, edge_index)
        x_relu3 = F.relu(x3)

        return x_relu3


def contrastive_loss(embeddings, edge_index, negative_ratio=10):
    embeddings = embeddings.to(device)
    edge_index = edge_index.to(device)
    pos_samples = embeddings[edge_index[0]] - embeddings[edge_index[1]]
    pos_distance = torch.sum(pos_samples ** 2, dim=1)

    neg_edge_index = negative_sampling(edge_index, num_nodes=embeddings.size(0),
                                       num_neg_samples=negative_ratio * edge_index.size(1))
    neg_samples = embeddings[neg_edge_index[0]] - embeddings[neg_edge_index[1]]
    neg_distance = torch.sum(neg_samples ** 2, dim=1)

    margin = 1
    neg_loss = torch.clamp(margin - neg_distance, min=0)
    loss = torch.mean(pos_distance) + torch.mean(neg_loss)
    return loss


drug_target_accociation_edgelist_new = pd.read_csv("output/drug_target_accociation_edgelist_new.csv", index_col=0)
filtered_total_pandas_renamed = pd.read_csv("output/filtered_GAE_total_pandas.csv", index_col=0)

# 将source和target列分别提取到两个Numpy数组中
sources = drug_target_accociation_edgelist_new['source'].to_numpy()
targets = drug_target_accociation_edgelist_new['target'].to_numpy()

# 将两个Numpy数组堆叠为一个(edge_num, 2)形状的数组
edges = np.stack((sources, targets), axis=1)

# 转换为PyTorch张量
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

print(edge_index.shape)

# 将Numpy数组转换为PyTorch张量
features_matrix = filtered_total_pandas_renamed.drop("node_id", axis=1).to_numpy()

feature_matrix_torch = torch.tensor(features_matrix, dtype=torch.float)

print(feature_matrix_torch.shape)

data = Data(x=feature_matrix_torch,
            edge_index=edge_index)
print(data.x)
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)

num_features = 128
num_classes = 2
hidden_channels1 = 256
hidden_channels2 = 128  # hidden_channel_number
hidden_channels3 = 64

# hidden_channel_number
GAT_epochs = 500000

model = GAT(num_features, hidden_channels1, hidden_channels2, hidden_channels3)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    embeddings = model(data)
    contrastive_loss_values = contrastive_loss(embeddings, data.edge_index)
    contrastive_loss_values.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return contrastive_loss_values


contrastive_loss_values_save = []

for epoch in range(1, GAT_epochs + 1):
    contrastive_loss_values = train()
    contrastive_loss_values_save.append(contrastive_loss_values)
    if epoch % 1000 == 0:
        print(f'Epoch: {epoch}, Contrastive Loss: {contrastive_loss_values}')

    # Check if epoch is a multiple of 10000 and get final_node_embeddings
    if epoch % 10000 == 0:
        model.eval()
        final_node_embeddings = model(data).detach().cpu().numpy()
        GAT_final_node_embeddings_pandas = pd.DataFrame(final_node_embeddings)
        GAT_final_node_embeddings_pandas = GAT_final_node_embeddings_pandas.reset_index()
        GAT_final_node_embeddings_pandas.to_csv(
            f"output1019/GAT_final_node_embeddings_pandas_{hidden_channels1}_{hidden_channels2}_{hidden_channels3}_{heads}_{epoch}_GATv2ConvGATconv.csv")
        model.train()  # Switch back to training mode

# At the end of the loop, final_node_embeddings will contain the embeddings of the last 10000th epoch.
ontrastive_loss_values_save = [tensor_loss.item() for tensor_loss in contrastive_loss_values_save]
import csv

# 打开一个新的CSV文件并写入
with open(f'output/loss {hidden_channels1} {hidden_channels2}_{hidden_channels3}_{heads}_{epoch}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for value in contrastive_loss_values_save:
        # 把每个值包装在一个列表中，并写入一行
        writer.writerow([value])

# 创建一个新的图形
plt.figure()

# 绘制损失曲线
plt.plot(contrastive_loss_values_save)

# 设置图形标题和坐标轴标签
plt.title("Contrastive Loss Curve for GAT training")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# 显示图形
plt.savefig(
    f"output/Contrastive Loss Curve for {hidden_channels1}_{hidden_channels2}_{hidden_channels3} GAT training_{heads}_{GAT_epochs}_GATv2ConvGATconv.png")

model.eval()
final_node_embeddings = model(data).detach().cpu().numpy()
GAT_final_node_embeddings_pandas = pd.DataFrame(final_node_embeddings)
GAT_final_node_embeddings_pandas = GAT_final_node_embeddings_pandas.reset_index()

GAT_final_node_embeddings_pandas.to_csv(
    f"output/GAT_final_node_embeddings_pandas_{hidden_channels1}_{hidden_channels2}_{hidden_channels3}_{heads}_{GAT_epochs}_GATv2ConvGATconv.csv")
