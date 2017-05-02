# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import pickle
import os,sys
import numpy as np
import sklearn

# import matplotlib.pyplot as plt
from entity.config import Config
from entity.data import DataProvider

from gensim.models.ldamodel import LdaModel
from gensim import corpora

from experiment.gensim_data import load_review_text

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


tags_of_interest = ['American(Traditional)', 'Italian', 'Mexican', 'Chinese', 'Japanese', 'Mediterranean', 'Indian', 'Thai', 'French', 'Greek']

def prepare_10class_dataset(dp, xy_path):
    '''
    Extract the data of interest
    :param dp:
    :param xy_path:
    :return:
    '''
    if os.path.exists(xy_path):
        with open(xy_path, 'rb') as f:
            X_idx, business_idx, Y, Y_name, Y_original = pickle.load(f)
    else:
        tag_dict = {}
        for t_id, tag in enumerate(dp.idx2tag):
            tag_dict[tag] = t_id

        tag_dict_of_interest = {}
        for t in tags_of_interest:
            tag_dict_of_interest[t] = tag_dict[t]

        X_idx = []
        Y = []
        Y_name = []
        Y_original = []
        print(tag_dict_of_interest)
        for doc_id, doc_tag_list in enumerate(dp.doc_tag_cor_fmatrix):
            # only keep the businesses that have only one label of interest
            class_id = -1
            count_found = 0
            for id, (t,t_id) in enumerate(tag_dict_of_interest.items()):
                # t_id is the original tag id, we map it into 0-9
                if doc_tag_list[t_id]:
                    count_found += 1
                    class_id = id
                    class_name = t
                    original_class_id = t_id
            if count_found == 1:
                X_idx.append(doc_id)
                Y.append(class_id)
                Y_name.append(class_name)
                Y_original.append(original_class_id)
        Y = np.asarray(Y)
        Y_name = np.asarray(Y_name)
        Y_original = np.asarray(Y_original)

        business_id_dict = {}
        for id_, business_id in enumerate(dp.idx2prod):
            business_id_dict[id_] = business_id
        business_idx = [business_id_dict[x_] for x_ in X_idx]

        with open(xy_path, 'wb') as f:
            pickle.dump([X_idx, business_idx, Y, Y_name, Y_original], f)

    return X_idx, business_idx, Y, Y_name, Y_original

model_name = 'doc2vec' # lda, ntm, doc2vec
model_dirs = {'lda':'ntm_model.freq=100.word=22548.lr=0.01', 'ntm':'ntm_model.freq=100.word=22548.lr=0.01', 'doc2vec':'doc2vec_model.freq=100'}

if __name__ == '__main__':
    print('Run experiment for %s' % model_name)
    ########################## Load the X-y pairs #############################
    print('Load the X-y pairs')
    args = sys.argv
    # args = [args[0], "prodx_doc2vec", "prod", "200", "5"]
    args = [args[0], model_dirs[model_name], "prod", "200", "5"]
    print(args)
    flag = args[1]
    n_processer = int(args[4])
    conf = Config(flag, args[2], int(args[3]))
    print(flag)

    home = os.environ["HOME"]
    home_path = home + "/Data/yelp/10-category-classification/"
    xy_path = home_path + 'yelp_10class_Xid_Y.pkl'

    dp = DataProvider(conf)
    doc_embed = np.load(conf.path_doc_npy+'.npy')
    word_embed = np.load(conf.path_word_npy+'.npy')
    model_weights = np.load(conf.path_model_npy+'.npy')

    # load the data points of interest
    X_idx, business_idx, Y, Y_name, Y_original = prepare_10class_dataset(dp, xy_path)
    ###################### Get the data samples #########################
    if model_name == 'ntm':
        X = doc_embed[0][X_idx]
    elif model_name == 'doc2vec':
        X = doc_embed[0][X_idx]
    elif model_name == 'lda':
        if os.path.exists(conf.path_lda_doc_vector):
            print('Loading LDA document vectors')
            with open(conf.path_lda_doc_vector, 'rb') as f:
                X = pickle.load(f)
        else:
            # get the doc_id in gensim corpus which correspond to the tags of interests
            gensim_business_document_dict      = load_review_text(conf)
            business_idx_dict           = {}
            for idx, busi_idx in enumerate(gensim_business_document_dict.keys()):
                business_idx_dict[busi_idx] = idx
            gensim_interested_idx      = [business_idx_dict[busi_id] for busi_id in business_idx]
            corpus = corpora.MmCorpus(conf.path_gensim_corpus)
            gensim_interested_docs = corpus[gensim_interested_idx]
            dictionary = corpora.Dictionary.load_from_text(conf.path_gensim_dict)
            with open(conf.lda_model_path, 'rb') as f:
                lda = pickle.load(f)
            # lda.print_topics(100)

            X = np.zeros((len(X_idx), conf.dim_item), dtype=float)
            for idx, d in enumerate(gensim_interested_docs):
                x = lda[d]
                for i,v in x:
                    X[idx][i] += v
                if idx % 1000 == 0:
                    print('Processing doc %d' % idx)

            print('Dumping LDA document vectors')
            with open(conf.path_lda_doc_vector, 'wb') as f:
                pickle.dump(X, f)

    data_len = len(X_idx)
    cut_value = int(0.8 * data_len)
    X_train = X[:cut_value]
    Y_train = Y[:cut_value]

    X_test = X[cut_value:]
    Y_test = Y[cut_value:]
    ###################### Rum SVM and evaluation #########################
    from sklearn import svm

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

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(Y_test, Y_pred)

    # Plot Precision-Recall curve
    # lw = 2
    # plt.clf()
    # plt.plot(recall[0], precision[0], lw=lw, color='navy',
    #          label='Precision-Recall curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(precision))
    # plt.legend(loc="lower left")
    # plt.show()
    #
    # # Plot Precision-Recall curve for each class
    # plt.clf()
    # plt.plot(recall, precision, color='gold', lw=lw,
    #          label='micro-average Precision-recall curve (area = {0:0.2f})'
    #                ''.format(precision))
    #
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Extension of Precision-Recall curve to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()