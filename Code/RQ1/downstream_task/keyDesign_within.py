# -*- coding:utf-8 -*-
import time
import os
import tensorflow as tf
import utils
from sklearn.model_selection import RepeatedKFold, train_test_split, StratifiedKFold  # 添加 train_test_split
import numpy as np
import ClassifierOutput
import pandas as pd
import secrets
from tqdm import tqdm

######################################################################################################################
baseURL = "/home/hyq2022/hyq/projects/CGCN-main-hyq/CGCN-keyDesign/dataset_keyDesign_Sora"
######################################################################################################################

# 设置
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
# 'LogisticRegression', 'DecisionTree', 'RandomForest', 'MLP'
flags.DEFINE_string('classifier', 'RandomForest', 'Select a classifier for classification.')
# 'SMOTE', 'SMOTETomek', 'underSample'
flags.DEFINE_string('imbalance', 'SMOTE', 'Select a methods of dealing with imbalanced data.')

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 训练并评估分类器模型
def run_evaluation(X_train, y_train, X_test, y_test):
    start_time = time.time()
    # 数据采样
    X_resampled, y_resampled = utils.generate_imbalance_data(X_train, y_train, FLAGS.imbalance)

    # 训练分类器并评估性能
    predprob_auc, predprob, precision, recall, f1, auc, mcc, accuracy, Brier_score = \
        ClassifierOutput.classifier_output(FLAGS.classifier, X_resampled, y_resampled, X_test, y_test,
                          grid_sear=True)  
    return precision, recall, f1, auc, accuracy, mcc, Brier_score

# 统计训练集和测试集关键类的个数和索引
def seed_data_record(y_train, y_test, train_index, test_index):
    # 关键类的标签值
    key_class_label = 1

    # 统计关键类的个数
    train_key_class_count = np.sum(y_train == key_class_label)
    test_key_class_count = np.sum(y_test == key_class_label)

    # 统计非关键类的个数
    train_non_key_class_count = len(y_train) - train_key_class_count
    test_non_key_class_count = len(y_test) - test_key_class_count

    # 关键类的索引
    train_key_class_indices = train_index[y_train == key_class_label]
    test_key_class_indices = test_index[y_test == key_class_label]

    return (train_key_class_count, train_non_key_class_count), (test_key_class_count, test_non_key_class_count), train_key_class_indices, test_key_class_indices

# 加载数据
def load_data(data):
    F1_list = []
    precision_list = []
    recall_list = []
    AUC_list = []
    accuracy_list = []
    mcc_list = []
    Brier_score_list = []

    # train_count_list = []
    # test_count_list = []
    # test_sample_counts = []
    # train_key_indices_list = []
    # test_key_indices_list = []
    # train_indices_list = []
    # test_indices_list = []

    # seeds_list = []

    # out_base_path = "/home/hyq2022/hyq/projects/CGCN-main-hyq/CGCN-keyDesign/eval_results/within100" 
    out_base_path = "/home/hyq2022/hyq/projects/CGCN-main-hyq/CGCN-keyDesign/eval_results/Sora_AST_result"
    out_path = os.path.join(out_base_path, "non_dw")
    os.makedirs(out_path, exist_ok=True)

    # 读取原始数据和处理后的数据
    origin_train_data = pd.read_csv(baseURL + "/" + data + "/Process-Binary.csv", header=0, index_col=False)
    dw_train_data = pd.read_csv(baseURL + "/" + data + "/" + "emb_AST_Sora_non-dw.csv", header=0, index_col=False)
    # X = np.array(pd.concat([dw_train_data, origin_train_data.iloc[:, 3:-1]], axis=1))
    X = np.array(dw_train_data)
    y = np.array(origin_train_data['KeyDesign'])
    
    # 设置重复次数
    n_repeats = 100
    
    # for repeat in range(n_repeats):
    for repeat in tqdm(range(n_repeats), desc=f"{data}", ascii=True):
        seed = secrets.randbelow(2**32)
        # seeds_list.extend([seed] * 2)  # 每个 repeat 内部的所有 fold 都使用相同的 seed

        # 使用 StratifiedKFold 进行 2 折交叉验证
        kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            precision, recall, fmeasure, auc, accuracy, mcc, Brier_score = run_evaluation(X_train, y_train, X_test, y_test)
            F1_list.append(fmeasure)
            precision_list.append(precision)
            recall_list.append(recall)
            AUC_list.append(auc)
            accuracy_list.append(accuracy)
            mcc_list.append(mcc)
            Brier_score_list.append(Brier_score)

    #         if repeat==0:
    #             train_counts, test_counts, train_key_indices, test_key_indices = seed_data_record(y_train, y_test, train_index, test_index)
    #             train_count_list.append(train_counts)
    #             test_count_list.append(test_counts)
    #             test_sample_counts.append(len(test_index))
    #             # train_key_indices_list.append(train_key_indices.tolist())
    #             test_key_indices_list.append(test_key_indices.tolist())
    #             # train_indices_list.append(train_index.tolist())
    #             test_indices_list.append(test_index.tolist())
    # if repeat==0:
    #     # 测试集信息保存结果到 CSV 文件
    #     results_path = "./unsupervised/test-index/Sora/Sora-k2/non_dw"
    #     os.makedirs(results_path, exist_ok=True)
    #     results_file = os.path.join(results_path, f"{data}_split_stats_2_splits.csv")

        # results_df = pd.DataFrame({
        #     'repeat': np.repeat(np.arange(n_repeats), 2),
        #     'fold': np.tile(np.arange(2), n_repeats),
        #     'seed': seeds_list,
        #     'train_key_class_count': [counts[0] for counts in train_count_list],
        #     'train_non_key_class_count': [counts[1] for counts in train_count_list],
        #     'test_key_class_count': [counts[0] for counts in test_count_list],
        #     'test_non_key_class_count': [counts[1] for counts in test_count_list],
        #     'test_sample_counts': test_sample_counts,
        #     # 'train_key_indices': train_key_indices_list,
        #     'test_key_indices': test_key_indices_list,
        #     # 'train_indices': train_indices_list,
        #     'test_indices': test_indices_list
        # })
        # results_df.to_csv(results_file, index=False)

    median = []
    median.append(np.median(precision_list))
    median.append(np.median(recall_list))
    median.append(np.median(F1_list))
    median.append(np.median(AUC_list))
    median.append(np.median(accuracy_list))
    median.append(np.median(mcc_list))
    median.append(np.median(Brier_score_list))

    avg = []
    avg.append(utils.average_value(precision_list))
    avg.append(utils.average_value(recall_list))
    avg.append(utils.average_value(F1_list))
    avg.append(utils.average_value(AUC_list))
    avg.append(utils.average_value(accuracy_list))
    avg.append(utils.average_value(mcc_list))
    avg.append(utils.average_value(Brier_score_list))

    # 对结果进行舍入到三位小数
    precision_list = [round(value, 4) for value in precision_list]
    recall_list = [round(value, 4) for value in recall_list]
    F1_list = [round(value, 4) for value in F1_list]
    AUC_list = [round(value, 4) for value in AUC_list]
    accuracy_list = [round(value, 4) for value in accuracy_list]
    mcc_list = [round(value, 4) for value in mcc_list]
    Brier_score_list = [round(value, 4) for value in Brier_score_list]
    median = [round(value, 4) for value in median]
    avg = [round(value, 4) for value in avg]

    name = ['precision', 'recall', 'F1', 'AUC', 'Accuracy', 'Mcc', 'Brier-score']
    results = [precision_list, recall_list, F1_list, AUC_list, accuracy_list, mcc_list, Brier_score_list]
    df = pd.DataFrame(data=results)
    df.index = name
    df.insert(0, 'median', median)
    df.insert(1, 'avg', avg)
    df.index.name = "metrics"
    # 创建目录，如果目录不存在
    os.makedirs(out_path, exist_ok=True)
    # 将结果写入 CSV 文件，确保路径指向文件的父目录而不是目录本身
    df.to_csv(out_path + "/" + data + "_Sora.csv")

def main():
    dict_file=open('/home/hyq2022/hyq/projects/CGCN-main-hyq/CGCN-keyDesign/configs/within.txt','r')
    lines=dict_file.readlines()
    for line in lines:
        # 加载数据并进行评估
        load_data(line.strip())

if __name__ == "__main__":
    main()
