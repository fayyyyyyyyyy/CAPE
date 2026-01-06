# -*- coding:utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB
import warnings
import utils
import numpy as np 
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore",category=ConvergenceWarning)
    from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# 根据指定的分类器名称，对输入的训练数据和测试数据进行分类器的训练和预测，返回评估指标
def classifier_output(classifier_name,data_train,label_train,data_test,label_test,grid_sear=False):
    weight_dict={0:1, 1:2}
    if(classifier_name=="LogisticRegression"):
        rf = LogisticRegression(class_weight=weight_dict, solver='liblinear')
        # 是否启用网格搜索(Grid Search)来寻找最佳的超参数组合
        if(grid_sear==False):
            rf.fit(data_train, label_train)
            predprob=rf.predict(data_test)
            predprob_auc=rf.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy,Brier_score
        if(grid_sear==True):
            # 正则化方式 (penalty)和正则化强度 C
            parameters = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
            # 对模型进行超参数调优                                 cv：交叉验证的折数
            gsearch = GridSearchCV(rf, parameters, scoring='f1', cv=5, n_jobs=8)
            gsearch.fit(data_train, label_train)
            predprob=gsearch.predict(data_test)
            predprob_auc=gsearch.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy,Brier_score
    elif classifier_name == "BNB":  # Bernoulli Naive Bayes
        rf = BernoulliNB()
        if(grid_sear==False):
            rf.fit(data_train, label_train)
            predprob = rf.predict(data_test)
            predprob_auc = rf.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy,Brier_score
    elif classifier_name == "DT":  # Decision Tree
        rf = DecisionTreeClassifier()
        if grid_sear == False:
            rf.fit(data_train, label_train)
            predprob = rf.predict(data_test)
            predprob_auc = rf.predict_proba(data_test)[:, 1]
            recall = metrics.recall_score(label_test, predprob)
            mcc = metrics.matthews_corrcoef(label_test, predprob)
            auc = metrics.roc_auc_score(label_test, predprob_auc)
            precision = metrics.precision_score(label_test, predprob)
            fmeasure = metrics.f1_score(label_test, predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc, predprob, precision, recall, fmeasure, auc, mcc, accuracy, Brier_score

    elif classifier_name == "GNB":  # Gaussian Naive Bayes
        rf = GaussianNB()
        if(grid_sear==False):
            rf.fit(data_train, label_train)
            predprob = rf.predict(data_test)
            predprob_auc = rf.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy,Brier_score
    elif classifier_name == "SVC":
        rf = SVC(probability=True, class_weight=weight_dict)  # SVM 需要 probability=True 才能输出概率
        if grid_sear:
            parameters = {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"]}
            gsearch = GridSearchCV(rf, parameters, scoring="f1", cv=5, n_jobs=8)
            gsearch.fit(data_train, label_train)
            rf = gsearch.best_estimator_
        rf.fit(data_train, label_train)
        predprob = rf.predict(data_test)
        predprob_auc = rf.predict_proba(data_test)[:, 1]
        predprob=rf.predict(data_test)
        predprob_auc=rf.predict_proba(data_test)[:, 1]
        recall=metrics.recall_score(label_test,predprob)
        mcc=metrics.matthews_corrcoef(label_test,predprob)
        auc=metrics.roc_auc_score(label_test,predprob_auc)
        precision=metrics.precision_score(label_test,predprob)
        fmeasure=metrics.f1_score(label_test,predprob)
        accuracy = metrics.accuracy_score(label_test, predprob)
        Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
        return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy,Brier_score
    
    elif(classifier_name=="DecisionTree"):
        rf = DecisionTreeClassifier(class_weight=weight_dict)
        if(grid_sear==False):
            rf.fit(data_train, label_train)
            predprob=rf.predict(data_test)
            predprob_auc=rf.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy,Brier_score
        if(grid_sear==True):
            parameters = {'criterion':['gini','entropy'],'max_depth':[30,50,60,100],'min_samples_leaf':[2,3,5,10],'min_impurity_decrease':[0.1,0.2,0.5]}
            gsearch = GridSearchCV(rf, parameters, scoring='f1', cv=5, n_jobs=8)
            gsearch.fit(data_train, label_train)
            predprob=gsearch.predict(data_test)
            predprob_auc=gsearch.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy,Brier_score

    elif(classifier_name=="RandomForest"):
        rf = RandomForestClassifier(class_weight=weight_dict)
        if(grid_sear==False):
            rf.fit(data_train, label_train)
            predprob=rf.predict(data_test)
            predprob_auc=rf.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy,Brier_score
        if(grid_sear==True):
            # print("Start Grid Search")
            # parameters = {'n_estimators':range(10,71,10), 'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20), 'min_samples_leaf':range(10,60,10)}
            parameters = {'n_estimators':range(10,71,10), 'max_depth':range(3,14,2), 'min_samples_split':range(10,201,20), 'min_samples_leaf':range(10,60,10)}
            gsearch = GridSearchCV(rf, parameters, scoring='f1', cv=3, n_jobs=14)
            gsearch.fit(data_train, label_train)
            predprob=gsearch.predict(data_test)
            predprob_auc=gsearch.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy,Brier_score
        
    elif(classifier_name=="MLP"):
        rf = MLPClassifier(random_state=10)
        if(grid_sear==False):
            rf.fit(data_train, label_train)
            predprob=rf.predict(data_test)
            predprob_auc=rf.predict_proba(data_test)[:, 1]  # 每个样本属于正类的概率
            recall=metrics.recall_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            f1=metrics.f1_score(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc,predprob,precision,recall,f1,auc,mcc,accuracy,Brier_score
        if(grid_sear==True):
            parameters = {"hidden_layer_sizes": [(100,), (100, 30)], "solver": ['adam', 'sgd'], "max_iter": [200, 400],}
            gsearch = GridSearchCV(rf, parameters, scoring='f1', cv=5, n_jobs=8)
            gsearch.fit(data_train, label_train)
            predprob=gsearch.predict(data_test) # 预测的类别标签
            predprob_auc=gsearch.predict_proba(data_test)[:, 1] # 预测的概率
            # print(predprob)
            # print(predprob_auc)
            recall=metrics.recall_score(label_test, predprob)
            precision=metrics.precision_score(label_test, predprob)
            f1=metrics.f1_score(label_test, predprob)
            mcc=metrics.matthews_corrcoef(label_test, predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            # Brier_score = metrics.brier_score_loss(label_test, predprob)
            auc=metrics.roc_auc_score(label_test, predprob_auc)
            # auc=utils.calculate_auc(label_test, predprob_auc)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc,predprob,precision,recall,f1,auc,mcc,accuracy,Brier_score
    elif classifier_name == "KNN":
        rf = KNeighborsClassifier()
        if grid_sear == False:
            rf.fit(data_train, label_train)
            predprob = rf.predict(data_test)
            predprob_auc = rf.predict_proba(data_test)[:, 1]
            recall = metrics.recall_score(label_test, predprob)
            mcc = metrics.matthews_corrcoef(label_test, predprob)
            auc = metrics.roc_auc_score(label_test, predprob_auc)
            precision = metrics.precision_score(label_test, predprob)
            fmeasure = metrics.f1_score(label_test, predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc, predprob, precision, recall, fmeasure, auc, mcc, accuracy, Brier_score
        if grid_sear == True:
            parameters = {'n_neighbors': list(range(1, min(11, data_train.shape[0] + 1))),
              'weights': ['uniform', 'distance']}
            gsearch = GridSearchCV(rf, parameters, scoring='f1', cv=3, n_jobs=8)
            gsearch.fit(data_train, label_train)
            predprob = gsearch.predict(data_test)
            predprob_auc = gsearch.predict_proba(data_test)[:, 1]
            recall = metrics.recall_score(label_test, predprob)
            auc = metrics.roc_auc_score(label_test, predprob_auc)
            precision = metrics.precision_score(label_test, predprob)
            fmeasure = metrics.f1_score(label_test, predprob)
            mcc = metrics.matthews_corrcoef(label_test, predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
            return predprob_auc, predprob, precision, recall, fmeasure, auc, mcc, accuracy, Brier_score

def find_best_threshold(label_test, predprob_auc):
    # 定义阈值范围
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0

    for threshold in thresholds:
        y_pred = (predprob_auc >= threshold).astype(int)
        precision = metrics.precision_score(label_test, y_pred)
        recall = metrics.recall_score(label_test, y_pred)
        f1 = metrics.f1_score(label_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_threshold, best_precision, best_recall, best_f1

def evaluate_mlp(classifier_name, data_train, label_train, data_test, label_test, grid_sear=True):
    
    if classifier_name == "MLP":
        rf = MLPClassifier(random_state=10)
        
        if grid_sear:
            parameters = {
                "hidden_layer_sizes": [(100,), (100, 30)],
                "solver": ['adam', 'sgd'],
                "max_iter": [200, 400],
            }
            gsearch = GridSearchCV(rf, parameters, scoring='f1', cv=5, n_jobs=8)
            gsearch.fit(data_train, label_train)
            predprob = gsearch.predict(data_test)  # 预测的类别标签
            predprob_auc = gsearch.predict_proba(data_test)[:, 1]  # 预测的概率
        else:
            rf.fit(data_train, label_train)
            predprob = rf.predict(data_test)  # 预测的类别标签
            predprob_auc = rf.predict_proba(data_test)[:, 1]  # 预测的概率
        
        # 找到最佳阈值
        best_threshold = 0.3
        # best_threshold, best_precision, best_recall, best_f1 = find_best_threshold(label_test, predprob_auc)
        print(f"best_threshold is {best_threshold}")
        # 使用最佳阈值进行预测
        predprob_optimized = (predprob_auc >= best_threshold).astype(int)
        
        recall = metrics.recall_score(label_test, predprob_optimized)
        precision = metrics.precision_score(label_test, predprob_optimized)
        f1 = metrics.f1_score(label_test, predprob_optimized)
        mcc = metrics.matthews_corrcoef(label_test, predprob_optimized)
        accuracy = metrics.accuracy_score(label_test, predprob_optimized)
        auc = utils.calculate_auc(label_test, predprob_auc)
        Brier_score = metrics.brier_score_loss(label_test, predprob_auc)
        
        return  predprob_auc,predprob,precision,recall,f1,auc,mcc,accuracy,Brier_score
