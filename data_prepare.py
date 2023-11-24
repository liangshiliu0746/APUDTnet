import numpy as np
import pandas as pd

drug_target_accociation_pandas = pd.read_csv(r"data/drugProtein.txt", sep="\t", header=None, index_col=None)
total_pandas = pd.read_csv("output/GAE_total_pandas.csv", index_col=0)


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


drug_similarity_data = generate_fused_drug_similarity_data("data")
number_of_drug_nodes = drug_similarity_data.shape[0]


def adjacency_matrix_to_edgelist(adjacency_matrix, difference):
    edgelist = pd.DataFrame(columns=["source", "target", "weight"])
    for i in range(0, adjacency_matrix.shape[0]):
        for j in range(0, adjacency_matrix.shape[1]):
            if (adjacency_matrix.iloc[i, j] != 0):
                try:
                    temp_number = adjacency_matrix.iloc[i, j]
                    int_temp_number = int(temp_number)
                    temp = pd.DataFrame(
                        {"source": [int(i)],
                         "target": [int(j + difference)],
                         "weight": [int_temp_number]})
                    edgelist = pd.concat([edgelist, temp], ignore_index=True)
                except Exception as e:
                    print(e)
    edgelist = edgelist.reset_index()
    edgelist = edgelist.drop("index", axis=1)
    return edgelist


drug_target_accociation_edgelist = adjacency_matrix_to_edgelist(drug_target_accociation_pandas, number_of_drug_nodes)

drug_list = drug_target_accociation_edgelist['source'].unique().tolist()
target_list = drug_target_accociation_edgelist['target'].unique().tolist()
total_list = [*drug_list, *target_list]
total_list.sort()
total_list_pandas = pd.DataFrame(total_list)
total_list_pandas = total_list_pandas.reset_index()
total_list_pandas = total_list_pandas.rename(columns={0: 'node_id'})
exist_nodeids = total_list_pandas['node_id'].unique().tolist()
filtered_total_pandas = total_pandas[total_pandas['node_id'].isin(exist_nodeids)]

mapping_dict = dict(zip(total_list_pandas["node_id"], total_list_pandas["index"]))
drug_target_accociation_edgelist_new = drug_target_accociation_edgelist.replace(mapping_dict)
filtered_total_pandas_renamed = filtered_total_pandas.replace(mapping_dict)
filtered_total_pandas_renamed = filtered_total_pandas_renamed.reset_index()
filtered_total_pandas_renamed = filtered_total_pandas_renamed.drop('index', axis=1)

drug_target_accociation_edgelist_new.to_csv("output/drug_target_accociation_edgelist_new.csv")
filtered_total_pandas_renamed.to_csv("output/filtered_GAE_total_pandas.csv")
