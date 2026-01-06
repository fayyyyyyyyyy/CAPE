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
# 设置随机种子
seed = 123
np.random.seed(seed)
# tf.random.set_seed(seed)

######################################################################################################################
baseURL = "/home/hyq2022/hyq/projects/CGCN-main-hyq/CGCN-keyDesign/dataset_keyDesign_FGCS"
######################################################################################################################
# 设置
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
# 'LogisticRegression', 'DecisionTree', 'RandomForest', 'MLP'
flags.DEFINE_string('classifier', 'RandomForest', 'Select a classifier for classification.')
# 'SMOTE', 'SMOTETomek', 'underSample'
flags.DEFINE_string('imbalance', 'SMOTE', 'Select a methods of dealing with imbalanced data.')

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.is_gpu_available())
