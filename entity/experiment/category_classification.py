# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import pickle
import os,sys
import numpy as np
import sklearn

from model.config import Config
from model.data import DataProvider


__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

args = sys.argv
if len(args) <= 10:
    # args = [args[0], "prodx_doc2vec", "prod", "200", "5"]
    args = [args[0], "prodx_sigmoid_softmax.lr=0.01", "prod", "200", "5"]
print(args)
flag = args[1]
n_processer = int(args[4])
conf = Config(flag, args[2], int(args[3]))
print(flag)

tags_of_interest = ['American(Traditional)', 'Italian', 'Mexican', 'Chinese', 'Japanese', 'Mediterranean', 'Indian', 'Thai', 'French', 'Greek']

def prepare_dataset(dp, xy_path):
    if os.path.exists(xy_path):
        with open(xy_path, 'rb') as f:
            X_idx, Y = pickle.load(f)
    else:
        tag_dict = {}
        for t_id, tag in enumerate(dp.idx2tag):
            tag_dict[tag] = t_id

        tag_dict_of_interest = {}
        for t in tags_of_interest:
            tag_dict_of_interest[t] = tag_dict[t]

        X_idx = []
        Y = []
        print(tag_dict_of_interest)
        for doc_id, doc_tag_list in enumerate(dp.doc_tag_cor_fmatrix):
            class_id = -1
            count_found = 0
            for id, (t,t_id) in enumerate(tag_dict_of_interest.items()):
                if doc_tag_list[t_id]:
                    count_found += 1
                    class_id = id
            if count_found == 1:
                X_idx.append(doc_id)
                Y.append(class_id)
        Y = np.asarray(Y)

        with open(xy_path, 'wb') as f:
            pickle.dump([X_idx, Y], f)

    return X_idx, Y

if __name__ == '__main__':
    home = os.environ["HOME"]
    home_path = home + "/Data/yelp/10-category-classification/"
    xy_path = home_path + 'yelp_10class_Xid_Y.pkl'

    dp = DataProvider(conf)
    doc_embed = np.load(conf.path_doc_npy+'.npy')
    word_embed = np.load(conf.path_word_npy+'.npy')
    model_weights = np.load(conf.path_model_npy+'.npy')

    X_idx, Y = prepare_dataset(dp, xy_path)

    X = doc_embed[0][X_idx]
    data_len = len(X_idx)
    cut_value = int(0.8 * data_len)
    X_train = X[:cut_value]
    Y_train = Y[:cut_value]

    X_test = X[cut_value:]
    Y_test = Y[cut_value:]

    from sklearn import svm

    # Micro: p=0.244700,r=0.244700,f-score=0.244700
    # Macro: p=0.066027,r=0.102407,f-score=0.066027
    # clf = svm.SVC() # C=1.0, kernel='rbf', degree=3

    # Accuracy=0.223940
    # Micro: p=0.224382, r=0.224382, f-score=0.224382
    # Macro: p=0.079280, r=0.100627, f-score=0.079280
    clf = svm.LinearSVC(C = 1.0, class_weight = None, dual = True, fit_intercept = True,
    intercept_scaling = 1, loss = 'squared_hinge', max_iter = 1000,
    multi_class = 'ovr', penalty = 'l2', random_state = None, tol = 0.0001,
    verbose = 0)

    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    f_micro = sklearn.metrics.f1_score(Y_test, Y_pred, average='micro')
    p_micro = sklearn.metrics.precision_score(Y_test, Y_pred, average='micro')
    r_micro = sklearn.metrics.recall_score(Y_test, Y_pred, average='micro')

    f_macro = sklearn.metrics.f1_score(Y_test, Y_pred, average='macro')
    p_macro = sklearn.metrics.precision_score(Y_test, Y_pred, average='macro')
    r_macro = sklearn.metrics.recall_score(Y_test, Y_pred, average='macro')


    accuracy = sklearn.metrics.accuracy_score(Y_test, Y_pred)

    print('Accuracy=%f' % accuracy)

    print('*' * 10 + ' Micro Score ' + '*' * 10)
    print('p=%f' % p_micro)
    print('r=%f' % r_micro)
    print('f-score=%f' % f_micro)

    print('*' * 10 + ' Macro Score ' + '*' * 10)
    print('p=%f' % p_macro)
    print('r=%f' % r_macro)
    print('f-score=%f' % f_macro)

    sklearn.metrics.precision_recall_curve(Y_test, Y_pred)