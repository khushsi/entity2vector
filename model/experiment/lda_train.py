# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
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


def load_review_text(X_yelp_idx):
    if os.path.exists(yelp_text_path):
        with open(yelp_text_path, 'r') as f:
            collection = json.load(f)
    else:
        business_dict = set()
        document_dict = {}
        for idx in X_yelp_idx:
            document_dict[idx] = ''
        with open(conf.path_data, 'r') as f:
            line_count = 0
            for l in f:
                line_count += 1
                if line_count % 10000 == 0:
                    print('line = %d' % line_count)

                # token[0]=business_id, token[2]=processed review text
                tokens = l.split('\t')
                if len(tokens) != 3:
                    continue

                b_id = tokens[0].lower().strip()
                # if b_id not in business_dict:
                #     business_dict.add(b_id)
                #     print('%d\t%s\t%s' % (len(business_dict), b_id, b_id in document_dict))

                if b_id in document_dict:
                    document_dict[idx] += tokens[2] + ' '
        collection = [document_dict[idx] for idx in X_yelp_idx]

        with open(yelp_text_path, 'w') as f:
            json.dump(collection, f)

    return collection

def preprocess_corpus(documents):
    if os.path.exists(lda_home_path+'yelp_10class_review.mm'):
        corpus = corpora.MmCorpus(lda_home_path+'yelp_10class_review.mm')
        dictionary = corpora.Dictionary.load_from_text(lda_home_path+'yelp_10class_review.dict')
    else:
        stoplist = stopwords.words('english')
        texts = [[word for word in document.lower().split() if word not in stoplist]
                      for document in documents]

         # remove words that appear only once
        from collections import defaultdict
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        texts = [[token for token in text if frequency[token] > 1]
                      for text in texts]

        dictionary = corpora.Dictionary(texts)
        dictionary.save_as_text(lda_home_path+'yelp_10class_review.dict')  # store the dictionary, for future reference
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize(lda_home_path+'yelp_10class_review.mm', corpus)

    return corpus, dictionary

if __name__ == '__main__':
    home = os.environ["HOME"]
    lda_home_path = home + "/Data/yelp/10-category-classification/"
    lda_xy_path = lda_home_path + 'yelp_10class_Xid_Y.pkl'
    yelp_text_path = lda_home_path + 'yelp_10class_review_text.pkl'
    lda_model_path = lda_home_path + 'yelp_10class_review.lda'

    dp = DataProvider(conf)
    print('Loading documents from yelp dataset')
    X_text      = load_review_text(dp.idx2prod)
    print('Preprocessing documents')
    X_corpus, dictionary    = preprocess_corpus(X_text)

    if os.path.exists(lda_model_path):
        lda = np.load(lda_home_path)
    else:
        lda = LdaModel(corpus=X_corpus, id2word=dictionary, num_topics=200, update_every=1, passes=1)  # train model
        np.pickle.dump(lda, lda_model_path)


    # print(lda[doc_bow])  # get topic probability distribution for a document
    # lda.update(corpus2)  # update the LDA model with additional documents
    # print(lda[doc_bow])
    #
    # lda = LdaModel(corpus, num_topics=50, alpha='auto', eval_every=5)