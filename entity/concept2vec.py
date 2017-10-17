# plain entity 2 vec model
import os
from keras.models import Model
from keras.layers import Input
from keras.layers.core import *
from keras.layers.embeddings import *
import entity
from keras.layers.merge import dot, concatenate, Dot, Concatenate
from entity.model.layers import *
from entity.data import DataProvider
from keras.callbacks import ModelCheckpoint
from entity.model.callbacks import *
import numpy as np
from entity.config import Config
from keras.optimizers import *
import numpy as np
import theano
import sys
import tensorflow as tf


def build_doc2vec_model(conf,dp):
    n_terms = len(dp.idx2word)

    # initialize parameters (embeddings)
    word_embed_data = np.array(dp.word_embed)
    item_embed_data = np.random.rand(dp.get_item_size(), conf.dim_word)
    print("finish data processing")

    # define model
    word_input = Input(shape=(1,), dtype ="int32", name ="word_idx")
    item_pos_input = Input(shape=(1,), dtype ="int32", name ="item_pos_idx")
    item_neg_input = Input(shape=(1,), dtype ="int32", name ="item_neg_idx")

    word_embed = Embedding(output_dim=conf.dim_word, input_dim=n_terms, input_length=1, name="word_embed",
                           weights=[word_embed_data], trainable=False)
    item_embed = Embedding(output_dim=conf.dim_word, input_dim=dp.get_item_size(), input_length=1, name="item_embed",
                           weights=[item_embed_data], trainable=True)

    word_embed_ = word_embed(word_input)
    item_pos_embed_ = item_embed(item_pos_input)
    item_neg_embed_ = item_embed(item_neg_input)

    word_flatten = Flatten()
    word_embed_ = word_flatten(word_embed_)

    item_pos_embed_ = Flatten()(item_pos_embed_)
    item_neg_embed_ = Flatten()(item_neg_embed_)

    pos_layer_ = Dot(axes=-1, normalize=False, name="pos_layer")([word_embed_, item_pos_embed_])
    neg_layer_ = Dot(axes=-1, normalize=False, name="neg_layer")([word_embed_, item_neg_embed_])
    merge_layer_ = Concatenate(axis=-1, name="merge_layer")([pos_layer_, neg_layer_])


    model = Model(input=[word_input, item_pos_input, item_neg_input], output=[merge_layer_, pos_layer_])

    def ranking_loss(y_true, y_pred):
        pos = y_pred[:,0]
        neg = y_pred[:,1]
        loss = K.maximum(0.5 + neg - pos, 0.0)
        return K.mean(loss) + 0 * y_true

    def dummy_loss(y_true, y_pred):
        # loss = K.max(y_pred) + 0 * y_true
        loss = y_pred + 0 * y_true
        return loss

    model.compile(optimizer=Adam(lr=0.01), loss = {'merge_layer' : ranking_loss, "pos_layer": dummy_loss}, loss_weights=[1, 0])

    print("finish model compiling")
    print(model.summary())

    return model, item_embed, word_embed


if __name__ == '__main__':
    args = sys.argv
    if len(args) <= 10:
        args = [args[0], "prodx_doc2vec", "prod", "200", "30"]
    print(args)
    flag = args[1]
    n_processer = int(args[4])

    os.environ['MKL_NUM_THREADS'] = str(n_processer)
    os.environ['GOTO_NUM_THREADS'] = str(n_processer)
    os.environ['OMP_NUM_THREADS'] = str(n_processer)
    # os.environ['THEANO_FLAGS'] = 'device=gpu,blas.ldflags=-lblas -lgfortran'
    #os.environ['THEANO_FLAGS'] = 'device=gpu'

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    conf = Config(flag, args[2], int(args[3]))
    print(flag)

    # get data
    dp = DataProvider(conf)

    if os.path.exists(conf.path_checkpoint):
        print("load previous checker")


    dp.generate_init()
    model, item_embed, word_embed = build_doc2vec_model(conf,dp)
    model.fit_generator(generator=dp.generate_data(batch_size=conf.batch_size, is_validate=False), nb_worker=1, pickle_safe=False,
                        nb_epoch=conf.n_epoch, steps_per_epoch=int(np.ceil(conf.sample_per_epoch/conf.batch_size)),
                        validation_data = dp.generate_data(batch_size=conf.batch_size, is_validate=True), validation_steps=1,
                        verbose=1, callbacks=[
                            my_checker_point(item_embed, word_embed, model, conf),
                            ModelCheckpoint(filepath=conf.path_checkpoint, verbose=1, save_best_only=True)
                        ])
