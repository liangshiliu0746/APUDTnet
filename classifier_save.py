import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
import catboost as cb
import pickle

model = []
total_edgelist_new = pd.read_csv("total_edgelist.csv")
GAT_final_node_embeddings_pandas = pd.read_csv("output/GAT_final_node_embeddings_pandas_128_64_32.csv",index_col=0)


def generate_sample(all_associations, random_seed):
    known_associations = all_associations.loc[all_associations['weight'] == 1]
    unknown_associations = all_associations.loc[all_associations['weight'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)
    sample_df = pd.concat([known_associations, random_negative])
    return sample_df


for i in range(1, 101):
    start_time = time.time()
    print(i)
    sample = generate_sample(total_edgelist_new, i)
    df = sample.merge(GAT_final_node_embeddings_pandas,
                      left_on='source',
                      right_on='index',
                      suffixes=('', '_source')).drop('index', axis=1)
    df = df.merge(GAT_final_node_embeddings_pandas,
                  left_on='target',
                  right_on='index',
                  suffixes=('_source', '_target')).drop('index', axis=1)
    df1 = df.drop("source", axis=1)
    df1 = df1.drop("target", axis=1)
    y = df1['weight'].astype(int)
    X = df1.drop("weight", axis=1)
    # 创建随机森林分类器
    rf_classifier = cb.CatBoostClassifier(iterations=5000,
                                          # learning_rate=0.01,
                                          # depth=10,
                                          logging_level='Silent',
                                          random_state=42)
    rf_classifier.fit(X, y)
    with open('classifier_save/%s.pkl' % ('model_{}'.format(i)), 'wb') as file:
        pickle.dump(rf_classifier, file)
    stop_time = time.time()
    total_time = stop_time - start_time
    print("本次运行时间为%d秒" % total_time)
    print(f"模型{i}保存完成")