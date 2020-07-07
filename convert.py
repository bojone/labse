#! -*- coding: utf-8 -*-
# require: tf 2.x

import tensorflow as tf
import tensorflow_hub as hub

labse_layer = hub.KerasLayer('/root/kg/bert/labse_old')

names = [
    'bert/embeddings/word_embeddings', 'bert/embeddings/position_embeddings',
    'bert/embeddings/token_type_embeddings', 'bert/embeddings/LayerNorm/gamma',
    'bert/embeddings/LayerNorm/beta'
]

for i in range(12):
    prefix = 'bert/encoder/layer_%d/' % i
    names.extend([
        prefix + 'attention/self/query/kernel',
        prefix + 'attention/self/query/bias',
        prefix + 'attention/self/key/kernel',
        prefix + 'attention/self/key/bias',
        prefix + 'attention/self/value/kernel',
        prefix + 'attention/self/value/bias',
        prefix + 'attention/output/dense/kernel',
        prefix + 'attention/output/dense/bias',
        prefix + 'attention/output/LayerNorm/gamma',
        prefix + 'attention/output/LayerNorm/beta',
        prefix + 'intermediate/dense/kernel',
        prefix + 'intermediate/dense/bias',
        prefix + 'output/dense/kernel',
        prefix + 'output/dense/bias',
        prefix + 'output/LayerNorm/gamma',
        prefix + 'output/LayerNorm/beta',
    ])

names.extend([
    'bert/pooler/dense/kernel',
    'bert/pooler/dense/bias',
])

weights = [w.numpy() for w in labse_layer.weights]


def create_variable(name, value):
    return tf.Variable(value, name=name)


with tf.Graph().as_default():

    for n, w in zip(names, weights):
        if w.shape[:2] == (12, 64):
            w = w.reshape((768,) + w.shape[2:])
        elif w.shape[-2:] == (12, 64):
            w = w.reshape(w.shape[:-2] + (768,))
        _ = create_variable(n, w)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        saver.save(
            sess, '/root/kg/bert/labse/bert_model.ckpt', write_meta_graph=False
        )
