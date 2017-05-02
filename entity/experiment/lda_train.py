# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import pickle
import os
import os,sys
import numpy as np
import sklearn
from gensim.models.ldamodel import LdaModel
from gensim import corpora

from nltk.corpus import stopwords

from entity.config import Config
from entity.data import DataProvider
from experiment.gensim_data import load_review_text, preprocess_corpus

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    args = sys.argv
    if len(args) <= 10:
        args = [args[0], "prodx_sigmoid_softmax", "prod", "200", "5"]
    print(args)
    flag = args[1]
    n_processer = int(args[4])
    conf = Config(flag, args[2], int(args[3]))
    print(flag)

    # home = os.environ["HOME"]
    # home_path = home + "/Data/yelp/"
    # data_path = home + "/Data/yelp/output/"
    # yelp_text_path = data_path + 'restaurant_review_pairs_freq=%d.txt' % conf.tf_cutoff
    # processed_document_path = data_path + 'restaurant_processed_document.txt'


    # dp = DataProvider(conf)
    print('Loading documents from yelp dataset')
    business_document_dict      = load_review_text(conf.path_review_pairs)
    business_idx     = business_document_dict.keys()
    documents        = business_document_dict.values()

    print('Preprocessing documents')
    X_corpus, dictionary         = preprocess_corpus(documents)

    if os.path.exists(conf.lda_model_path):
        with open(conf.lda_model_path, 'rb') as f:
            lda = pickle.load(f)
    else:
        lda = LdaModel(corpus=X_corpus, id2word=dictionary, num_topics=200, update_every=1, passes=1)  # train model
        with open(conf.lda_model_path, 'wb') as f:
            pickle.dump(lda, f)


    # print(lda[doc_bow])  # get topic probability distribution for a document
    # lda.update(corpus2)  # update the LDA model with additional documents
    # print(lda[doc_bow])
    #
    # lda = LdaModel(corpus, num_topics=50, alpha='auto', eval_every=5)