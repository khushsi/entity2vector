# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import os

from gensim import corpora
from nltk.corpus import stopwords

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def load_review_text(conf):
    yelp_text_path = conf.path_review_pairs
    if os.path.exists(conf.processed_document_path):
        with open(conf.processed_document_path, 'r') as f:
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

        with open(conf.processed_document_path, 'w') as f:
            json.dump(business_document_dict, f)

    print('Load reviews of %d businesses' % len(business_document_dict))
    return business_document_dict

def preprocess_corpus(conf, documents):
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
        texts = [[token for token in text if frequency[token] > conf.path_gensim_tf]
                      for text in texts]

        print('Exporting dict and corpus')
        dictionary = corpora.Dictionary(texts)
        dictionary.save_as_text(conf.path_gensim_dict)  # store the dictionary, for future reference
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize(conf.path_gensim_corpus, corpus)

    return corpus, dictionary
