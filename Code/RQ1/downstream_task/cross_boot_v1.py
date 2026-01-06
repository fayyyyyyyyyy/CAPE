# -*- coding:utf-8 -*-
import time
import os
import tensorflow as tf
import tqdm
import utils
import numpy as np
import ClassifierOutput
import pandas as pd
import secrets

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
# 'LogisticRegression', 'DecisionTree', 'RandomForest', 'MLP'
flags.DEFINE_string('classifier', 'MLP', 'Select a classifier for classification.')
# 'SMOTE', 'SMOTETomek', 'underSample'(RandomUnderSampler)
flags.DEFINE_string('imbalance', 'underSample', 'Select a methods of dealing with imbalanced data.')

# 训练并评估分类器模型
def run_evaluation(X_train, y_train, X_test, y_test, random_seed):
    start_time = time.time()
    # 数据采样
    X_resampled, y_resampled = utils.generate_imbalance_data(X_train, y_train, FLAGS.imbalance, random_seed)
    # 训练分类器并评估性能
    predprob_auc, predprob, precision, recall, f1, auc, mcc, accuracy, Brier_score = \
        ClassifierOutput.classifier_output(FLAGS.classifier, X_resampled, y_resampled, X_test, y_test,
                          grid_sear=True)  
    return precision, recall, f1, auc, accuracy, mcc, Brier_score


# cross-version/cross-project: setting1
def load_train_test(baseURL, datalist, csvfile):
    # loading embedding matrix and labels
    # The first version is used as training set
    origin_train_data = pd.read_csv(baseURL+datalist[0]+"/Process-Binary.csv", header=0, index_col=False)
    dw_train_data = pd.read_csv(baseURL+datalist[0]+"/"+ csvfile, header=0, index_col=False)
    X_train = np.array(dw_train_data)
    y_train = np.array(origin_train_data['KeyDesign'])

    # # 合并特征和标签
    # train_data = pd.concat([dw_train_data.reset_index(drop=True), origin_train_data[['KeyDesign']].reset_index(drop=True)], axis=1)
    # # train_data.to_csv('./cross_results/merged_train_data.csv', index=False)
    
    # # 有放回抽样：params-1:数据集的行数（样本数量）、2：抽样的样本数量，等于原始数据集的行数、3：表示有放回抽样。
    # boot = np.random.choice(train_data.shape[0], train_data.shape[0], replace=True)
    # train_data_sampled = train_data.iloc[boot].reset_index(drop=True)
    # # print("抽取的索引值:", boot)
    # # print("索引值长度:", len(boot))

    # hgg

    # The second version is used as test set
    origin_test_data = pd.read_csv(baseURL + datalist[1] + "/Process-Binary.csv", header=0, index_col=False)
    dw_test_data = pd.read_csv(baseURL + datalist[1] + "/" + csvfile, header=0, index_col=False)
    X_test = np.array(dw_test_data)
    # X_test = np.array(pd.concat([dw_test_data, origin_test_data.iloc[:, 3:-1]], axis=1))
    y_test = np.array(origin_test_data['KeyDesign'])
    return X_train, y_train, X_test, y_test

def metrics_csv(X_train, y_train, X_test, y_test, save_csv_path, datalist, random_seed):
    precision, recall, f1, auc, accuracy, mcc, Brier_score = run_evaluation(X_train, y_train, X_test, y_test, random_seed)
    precision = "{:.4f}".format(precision)
    recall = "{:.4f}".format(recall)
    f1 = "{:.4f}".format(f1)
    auc = "{:.4f}".format(auc)
    accuracy = "{:.4f}".format(accuracy)
    mcc = "{:.4f}".format(mcc)
    Brier_score = "{:.4f}".format(Brier_score)
    result = [datalist[0], datalist[1], precision, recall, f1, auc, accuracy, mcc, Brier_score]
    file_exists = os.path.exists(save_csv_path) and os.path.getsize(save_csv_path) > 0
    column_names = ['train', 'test', 'precision', 'recall', 'f1', 'auc', 'accuracy', 'mcc', 'Brier_score']
    df = pd.DataFrame([result])
    df.columns = column_names
    # 在文件为空或不存在时写入列名，否则只写入数据
    df.to_csv(save_csv_path, mode='a', header=not file_exists,index=False)

# loop eight projects
projects = ['FGCS', 'Kang', 'Sora']

baseURL = f'./dataset_keyDesign_FGCS/'
outputURL = './cross_results/source_results_demo-v1'

outputPath = os.path.join(outputURL, "FGCS", f'FGCS')
os.makedirs(outputPath, exist_ok=True)
# csvfiles=[f'CGCN_emb_FGCS_directed_ins.csv', f'CGCN_emb_FGCS_directed_outs.csv',
#         f'CGCN_emb_FGCS_dw_ins.csv', f'CGCN_emb_FGCS_dw_outs.csv',
#         f'CGCN_emb_FGCS_non_dw.csv', f'CGCN_emb_FGCS_onlyWeight.csv']
csvfiles=[f'CGCN_emb_FGCS_directed_ins.csv']
n_repeats = 100
    
for repeat in range(n_repeats):
# for repeat in tqdm(range(n_repeats), desc=f"FGCS", ascii=True):
    for csvfile in csvfiles:
        # 提取字符串substring：directed_ins、directed_outs、dw_ins、dw_outs、non_dw、onlyWeight
        start_index = csvfile.find(f'CGCN_emb_FGCS_') + len(f'CGCN_emb_FGCS_')    # 找到'CGCN_emb_FGCS_'之后的位置  
        end_index = csvfile.find('.csv')    # 找到'.csv'之前的位置
        # 提取子字符串
        substring = csvfile[start_index:end_index]

        dict_file=open('./configs/cross_project_demo.txt','r')   # 训练集和测试集的划分
        output_path = os.path.join(outputPath, f'{substring}-cross_result.csv')
        lines=dict_file.readlines()
        for line in lines:
            datalist = line.strip().split(',')
            random_seed = secrets.randbelow(2**32)
            print(line.strip() + " Start!")
            # print(datalist[0], datalist[-1])
            # output_path = os.path.join('./cross_results', f'{datalist[0]}-{datalist[-1]}.csv')
            X_train, y_train, X_test, y_test = load_train_test(baseURL, datalist, csvfile)
            metrics_csv(X_train, y_train, X_test, y_test, output_path, datalist,random_seed)

