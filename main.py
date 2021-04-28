import random
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split,StratifiedKFold


seed =2021



def load_vec(filename):
    line_number = 1
    node_vec = {}
    for i in open(filename).readlines():
        newi = i.strip().split(' ')
        node_vec[str(line_number)] = newi[:]
        line_number += 1
    return node_vec


def load_mydata():
    drug_vec = load_vec("feature/drug_feature.txt")
    protein_vec = load_vec("feature/protein_feature.txt")
    fw1 = open("data_of_drug_protein_interaction_half_cross", "w")
    fw2 = open("data_of_drug_protein_interaction_all_cross", "w")
    fw3 = open("data_of_drug_protein_interaction_no_cross", "w")
    for i in open("result_half_cross.txt").readlines():
        newi = i.strip().split('\t')
        vec1 = drug_vec.get(newi[0])
        vec2 = protein_vec.get(newi[1])
        if vec1 is not None and vec2 is not None:
            S = ""
            for vec in vec1:
                S += vec + "\t"
            for vec in vec2:
                S += vec + "\t"
            S += newi[2] + "\n"
            fw1.write(S)
    for i in open("result_all_cross.txt").readlines():
        newi = i.strip().split('\t')
        vec1 = drug_vec.get(newi[0])
        vec2 = protein_vec.get(newi[1])
        if vec1 is not None and vec2 is not None:
            S = ""
            for vec in vec1:
                S += vec + "\t"
            for vec in vec2:
                S += vec + "\t"
            S += newi[2] + "\n"
            fw2.write(S)
    for i in open("result_no_cross.txt").readlines():
        newi = i.strip().split('\t')
        vec1 = drug_vec.get(newi[0])
        vec2 = protein_vec.get(newi[1])
        if vec1 is not None and vec2 is not None:
            S = ""
            for vec in vec1:
                S += vec + "\t"
            for vec in vec2:
                S += vec + "\t"
            S += newi[2] + "\n"
            fw3.write(S)
    fw1.flush()
    fw1.close()
    fw2.flush()
    fw2.close()
    fw3.flush()
    fw3.close()

def load_data():
    X_pos=[]
    X_neg_all_cross=[]
    X_neg_no_cross=[]
    X_neg_half_cross=[]
    X=[]
    Y1_pos=[]
    Y1_neg_all_cross=[]
    Y1_neg_no_cross=[]
    Y1_neg_half_cross=[]
    Y1=[]

    for i in open("data_of_drug_protein_interaction_no_cross").readlines():
        newi=i.strip().split('\t')
        if int(newi[-1]) == 1:
            X_pos.append([float(x) for x in newi[:-1]])
            Y1_pos.append(int(newi[-1]))
        else:
            X_neg_no_cross.append([float(x) for x in newi[:-1]])
            Y1_neg_no_cross.append(int(newi[-1]))
            
    
    for i in open("data_of_drug_protein_interaction_all_cross").readlines():
        newi=i.strip().split('\t')
        if int(newi[-1]) == 0:
            X_neg_all_cross.append([float(x) for x in newi[:-1]])
            Y1_neg_all_cross.append(int(newi[-1]))
            
    for i in open("data_of_drug_protein_interaction_half_cross").readlines():
        newi=i.strip().split('\t')
        if int(newi[-1]) == 0:
            X_neg_half_cross.append([float(x) for x in newi[:-1]])
            Y1_neg_half_cross.append(int(newi[-1]))
    
    X = X_pos
    Y1 = Y1_pos
    
    random.seed(seed)

    for idx in random.sample(range(0,len(X_neg_all_cross)-1),1316):# positive_number * 26% = 1316
        X.append(X_neg_all_cross[idx])
        Y1.append(0)
    for idx in random.sample(range(0,len(X_neg_half_cross)-1),2809):# positive_number * 57% = 2809
        X.append(X_neg_half_cross[idx])
        Y1.append(0)
    for idx in random.sample(range(0,len(X_neg_no_cross)-1),853):# positive_number * 17% = 853
        X.append(X_neg_no_cross[idx])
        Y1.append(0)
        
    return np.array(X),np.array(Y1)

if __name__=="__main__":

    load_mydata()
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }  
    
    X,Y=load_data()
    
    kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0)
    
    test_auc_fold = []
    test_aupr_fold = []

    for train_index, test_index in kf.split(X, Y):
        X_train, Y_train = X[train_index,:], Y[train_index]
        X_test, Y_test = X[test_index,:], Y[test_index]
        train_data = lgb.Dataset(X_train, Y_train)
        eval_data = lgb.Dataset(X_test, Y_test)
        
        gbm = lgb.train(params, train_data, num_boost_round=3000, valid_sets=eval_data, early_stopping_rounds=200)
        
        Y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        
        test_auc = roc_auc_score(Y_test, Y_pred)
        test_aupr = average_precision_score(Y_test, Y_pred)
        
        test_auc_fold.append(test_auc)
        test_aupr_fold.append(test_aupr)
        
    
    test_auc_fold.append(np.mean(test_auc_fold))
    test_aupr_fold.append(np.mean(test_aupr_fold))
    
    print('AUROC:')
    print(test_auc_fold[:-1])
    print('AverageAUC:')
    print(test_auc_fold[-1])
    print('AUPR:')
    print(test_aupr_fold[:-1])
    print('AverageAUPR')
    print(test_aupr_fold[-1])
