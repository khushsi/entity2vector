# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from model.config import Config
from model_e2v_ntm import build_ntm_model
from model.data import DataProvider
import tensorflow as tf

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

with tf.device('/cpu:0'):
    args = sys.argv
    if len(args) <= 10:
        args = [args[0], "ntm_sigmoid_addneg", "prod", "200", "5"]
    print(args)
    flag = args[1]
    n_processer = int(args[4])
    conf = Config(flag, args[2], int(args[3]))
    print(flag)

    dp = DataProvider(conf)
    model, word_embed, item_embed = build_ntm_model(dp)
    model.load_weights(conf.path_checkpoint)

    print(model.summary())