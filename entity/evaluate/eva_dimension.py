from config import Config
from entity.data import DataProvider
from gensim.models.word2vec import Word2Vec
import numpy as np
import os
from scipy.stats import logistic
from entity.model_e2v_ntm import build_ntm_model
import tensorflow as tf

flag = "ntm_model.freq=100.word=22548.lr=0.01"
conf = Config(flag, "prod" , 200)

# wrong!
# model = np.load(conf.path_model_npy + ".npy")
# word_embed = model[0]
# prod_embed = model[1]
# transfer_w = model[2]
# transfer_b = model[3]

dp = DataProvider(conf)

print('Start loading data')
dp = DataProvider(conf)
print('Data load complete')
model, word_embed, item_embed = build_ntm_model(conf, dp)
print('Start loading model weights')
model.load_weights(conf.path_checkpoint)
print('Loading model weights complete')
prod_embed = model.weights[0]
transfer_w = model.weights[1]
transfer_b = model.weights[2]
word_embed = model.weights[3]

init_op = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init_op)
# weight = np.dot(word_embed, transfer_w)
weight = (tf.matmul(word_embed, transfer_w) + transfer_b).eval()
weight = logistic.cdf(weight)

for topic_id in range(conf.dim_item):
    word_probs = weight[:, topic_id]
    top_word_ids    = np.argsort(word_probs)[::-1][:50]
    top_word_probs  = np.sort(word_probs)[::-1][:50]
    # words = [(dp.idx2word[word_id], weight[word_id, topic_id]) for word_id in word_probs if weight[word_id, topic_id] > .9]
    word_probs = [(dp.idx2word[word_id], word_prob) for word_id, word_prob in zip(top_word_ids, top_word_probs)]
    print("Topic", topic_id)
    print(word_probs)
    print("=========================\n")

print("finish")







