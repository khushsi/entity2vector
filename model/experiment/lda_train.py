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

from model.config import Config
from model.data import DataProvider
from model.experiment.category_classification import prepare_dataset

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

args = sys.argv
if len(args) <= 10:
    args = [args[0], "prodx_sigmoid_softmax", "prod", "200", "5"]
print(args)
flag = args[1]
n_processer = int(args[4])
conf = Config(flag, args[2], int(args[3]))
print(flag)


def load_review_text(yelp_text_path):
    if os.path.exists(processed_document_path):
        with open(processed_document_path, 'r') as f:
            business_document_dict = json.load(f)
    else:
        business_document_dict = {} # to combine duplicated business
        with open(yelp_text_path, 'r') as f:
            line_count = 0
            for l in f:
                line_count += 1
                if line_count % 10000 == 0:
                    print('line = %d' % line_count)

                # token[0]=business_id, token[2]=processed review text
                tokens = l.split('\t')
                if len(tokens) != 3:
                    continue

                b_id = tokens[0].strip()
                # if b_id not in business_dict:
                #     business_dict.add(b_id)
                #     print('%d\t%s\t%s' % (len(business_dict), b_id, b_id in document_dict))

                business_document_dict[b_id] = business_document_dict.get(b_id, '') + ' ' + tokens[2]

        with open(processed_document_path, 'w') as f:
            json.dump(business_document_dict, f)

    print('Load reviews of %d businesses' % len(business_document_dict))
    return business_document_dict.keys(), business_document_dict.values()

def preprocess_corpus(documents):
    if os.path.exists(conf.path_gensim_corpus):
        corpus = corpora.MmCorpus(conf.path_gensim_corpus)
        dictionary = corpora.Dictionary.load_from_text(conf.path_gensim_dict)
    else:
        stoplist = stopwords.words('english')

        print('Removing stop words')
        texts = [[word for word in document.lower().split() if word not in stoplist]
                      for document in documents]

        print('Counting term frequency')
         # remove words that appear only once
        from collections import defaultdict
        frequency = defaultdict(int)

        for text in texts:
            for token in text:
                frequency[token] += 1

        print('Removing low tf terms')
        texts = [[token for token in text if frequency[token] > 1]
                      for text in texts]

        print('Exporting dict and corpus')
        dictionary = corpora.Dictionary(texts)
        dictionary.save_as_text(conf.path_gensim_dict)  # store the dictionary, for future reference
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize(conf.path_gensim_corpus, corpus)

    return corpus, dictionary

if __name__ == '__main__':
    home = os.environ["HOME"]
    home_path = home + "/Data/yelp/"
    data_path = home + "/Data/yelp/output/"
    yelp_text_path = data_path + 'restaurant_review_pairs_freq=%d.txt' % conf.tf_cutoff
    processed_document_path = data_path + 'restaurant_processed_document.txt'
    lda_model_path = home_path + 'model/lda/yelp_restaurant_review.lda'

    # dp = DataProvider(conf)
    print('Loading documents from yelp dataset')
    business_idx, documents      = load_review_text(yelp_text_path)

    print('Preprocessing documents')
    X_corpus, dictionary         = preprocess_corpus(documents)

    if os.path.exists(lda_model_path):
        with open(lda_model_path, 'rb') as f:
            lda = pickle.load(f)
    else:
        lda = LdaModel(corpus=X_corpus, id2word=dictionary, num_topics=200, update_every=1, passes=1)  # train model
        with open(lda_model_path, 'wb') as f:
            pickle.dump(lda, lda_model_path)


    # print(lda[doc_bow])  # get topic probability distribution for a document
    # lda.update(corpus2)  # update the LDA model with additional documents
    # print(lda[doc_bow])
    #
    # lda = LdaModel(corpus, num_topics=50, alpha='auto', eval_every=5)